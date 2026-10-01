//! A deliberately small Vulkan setup: enough to put a triangle on the screen
//! and no more, so there is room to experiment.
//!
//! Every call matches the Vulkan spec one for one, so any Vulkan reference
//! applies here directly. What is missing on purpose, and worth adding as you
//! go: vertex and index buffers, uniform buffers and descriptor sets, depth
//! testing, several frames in flight, and validation layers.
//!
//! # On macOS
//!
//! There is no native Vulkan here. MoltenVK translates to Metal, which the
//! loader finds through `VK_ICD_FILENAMES` — set for you in
//! `.cargo/config.toml`. Two things follow: the instance is created with the
//! portability-enumeration flag, and the device enables
//! `VK_KHR_portability_subset` when the driver reports it. Both are required by
//! the spec when talking to a non-conformant implementation, and both are the
//! usual first stumble on a Mac.
//!
//! # Safety
//!
//! Nearly every Vulkan call is `unsafe`: the API trusts the caller to pass
//! valid handles, to destroy things in the right order, and not to free
//! anything the GPU is still reading. The rules being followed here are that
//! objects are destroyed in reverse order of creation, and that
//! [`Renderer::destroy`] waits for the device to go idle first.

use ash::{Device, Entry, Instance, khr, vk};
use std::ffi::CStr;
use std::sync::Arc;
use winit::window::Window;

/// The compiled shaders, built from `shaders/` by `build.rs`.
const VERTEX_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.vert.spv"));
const FRAGMENT_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.frag.spv"));

/// Everything needed to draw, in creation order.
///
/// The fields are public so that experiments can reach in without this file
/// having to grow an accessor for each one.
pub struct Renderer {
    pub window: Arc<Window>,

    /// Holds the loaded Vulkan library open. Nothing reads it after start-up,
    /// but dropping it would unload the loader out from under every handle
    /// below, so it is kept for its lifetime rather than its value.
    #[allow(dead_code)]
    pub entry: Entry,
    pub instance: Instance,

    pub surface_api: khr::surface::Instance,
    pub surface: vk::SurfaceKHR,

    pub physical_device: vk::PhysicalDevice,
    /// Kept for experiments that want a second command pool or a transfer
    /// queue from the same family.
    #[allow(dead_code)]
    pub queue_family: u32,
    pub device: Device,
    pub queue: vk::Queue,

    pub swapchain_api: khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub views: Vec<vk::ImageView>,
    pub framebuffers: Vec<vk::Framebuffer>,

    pub render_pass: vk::RenderPass,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,

    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,

    /// One frame in flight, which is the least sync that is still correct: the
    /// fence makes the next frame wait for this one to finish.
    pub image_available: vk::Semaphore,
    pub render_finished: Vec<vk::Semaphore>,
    pub in_flight: vk::Fence,

    /// What the screen is cleared to. Change it and watch the window.
    pub clear_colour: [f32; 4],
}

impl Renderer {
    /// Brings Vulkan up for this window.
    ///
    /// # Panics
    ///
    /// On any Vulkan failure. This is a playground: a panic with the failing
    /// call in the message is more useful than an error type to thread through
    /// every line.
    pub fn new(window: Arc<Window>) -> Self {
        let entry: Entry = unsafe { Entry::load() }.expect(
            "no Vulkan loader. On macOS this means MoltenVK is missing:\n    \
             brew install molten-vk vulkan-loader",
        );

        let (instance, surface_api, surface) = create_instance_and_surface(&entry, &window);
        let (physical_device, queue_family) = pick_device(&instance, &surface_api, surface);
        let (device, queue) = create_device(&instance, physical_device, queue_family);

        let swapchain_api: khr::swapchain::Device = khr::swapchain::Device::new(&instance, &device);
        let (format, extent) = surface_settings(&surface_api, physical_device, surface, &window);
        let render_pass: vk::RenderPass = create_render_pass(&device, format);
        let (pipeline_layout, pipeline) = create_pipeline(&device, render_pass);

        let command_pool: vk::CommandPool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .expect("create_command_pool");

        let command_buffer: vk::CommandBuffer = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .expect("allocate_command_buffers")[0];

        let image_available: vk::Semaphore =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                .expect("create_semaphore");
        let in_flight: vk::Fence = unsafe {
            device.create_fence(
                // Signalled, so the first frame does not wait for a frame that
                // never happened.
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
        }
        .expect("create_fence");

        let mut renderer = Self {
            window,
            entry,
            instance,
            surface_api,
            surface,
            physical_device,
            queue_family,
            device,
            queue,
            swapchain_api,
            swapchain: vk::SwapchainKHR::null(),
            format,
            extent,
            views: Vec::new(),
            framebuffers: Vec::new(),
            render_pass,
            pipeline_layout,
            pipeline,
            command_pool,
            command_buffer,
            image_available,
            render_finished: Vec::new(),
            in_flight,
            clear_colour: [0.02, 0.02, 0.06, 1.0],
        };

        renderer.build_swapchain();

        renderer
    }

    /// Draws one frame.
    ///
    /// Returns without drawing when the window has no area, which is what a
    /// minimised window reports, and rebuilds the swapchain when the surface
    /// says it no longer matches — after a resize, or a move to a screen with a
    /// different scale.
    pub fn draw(&mut self) {
        if self.extent.width == 0 || self.extent.height == 0 {
            return;
        }

        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight], true, u64::MAX)
                .expect("wait_for_fences");
        }

        let acquired = unsafe {
            self.swapchain_api.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available,
                vk::Fence::null(),
            )
        };

        let index: u32 = match acquired {
            Ok((index, _suboptimal)) => index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.rebuild_swapchain();
                return;
            }
            Err(error) => panic!("acquire_next_image: {error:?}"),
        };

        unsafe {
            self.device
                .reset_fences(&[self.in_flight])
                .expect("reset_fences");
        }

        self.record(index);

        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [self.command_buffer];
        let wait = [self.image_available];
        // One semaphore per image: signalling the same one while a previous
        // present may still be waiting on it is the classic Vulkan hazard here.
        let signal = [self.render_finished[index as usize]];

        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal);

        unsafe {
            self.device
                .queue_submit(self.queue, &[submit], self.in_flight)
                .expect("queue_submit");
        }

        let swapchains = [self.swapchain];
        let indices = [index];
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal)
            .swapchains(&swapchains)
            .image_indices(&indices);

        let presented = unsafe { self.swapchain_api.queue_present(self.queue, &present) };

        match presented {
            Ok(false) => {}
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.rebuild_swapchain(),
            Err(error) => panic!("queue_present: {error:?}"),
        }
    }

    /// Notes a new window size; the swapchain itself is rebuilt on the next
    /// frame, since a resize arrives many times while a window is dragged.
    pub fn resized(&mut self) {
        self.rebuild_swapchain();
    }

    /// Writes the commands for one frame.
    ///
    /// The place to experiment: bind another pipeline, draw more, or push
    /// constants before the draw.
    fn record(&self, index: u32) {
        let command_buffer: vk::CommandBuffer = self.command_buffer;

        unsafe {
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("reset_command_buffer");

            self.device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                .expect("begin_command_buffer");

            let clear = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: self.clear_colour,
                },
            }];

            self.device.cmd_begin_render_pass(
                command_buffer,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffers[index as usize])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: self.extent,
                    })
                    .clear_values(&clear),
                vk::SubpassContents::INLINE,
            );

            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            // Viewport and scissor are dynamic, so a resize needs no new
            // pipeline.
            self.device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: self.extent.width as f32,
                    height: self.extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            self.device.cmd_set_scissor(
                command_buffer,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.extent,
                }],
            );

            // Three vertices, one instance. The corners live in the shader.
            self.device.cmd_draw(command_buffer, 3, 1, 0, 0);

            self.device.cmd_end_render_pass(command_buffer);
            self.device
                .end_command_buffer(command_buffer)
                .expect("end_command_buffer");
        }
    }

    /// Tears the swapchain down and builds it again at the window's new size.
    fn rebuild_swapchain(&mut self) {
        unsafe { self.device.device_wait_idle() }.expect("device_wait_idle");

        self.destroy_swapchain();

        let (format, extent) = surface_settings(
            &self.surface_api,
            self.physical_device,
            self.surface,
            &self.window,
        );
        self.format = format;
        self.extent = extent;

        if extent.width > 0 && extent.height > 0 {
            self.build_swapchain();
        }
    }

    /// Creates the swapchain, its views, its framebuffers and its semaphores.
    fn build_swapchain(&mut self) {
        let capabilities = unsafe {
            self.surface_api
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
        }
        .expect("get_physical_device_surface_capabilities");

        // One more than the minimum lets the driver work on the next image
        // while one is being shown, up to whatever maximum it declares.
        let wanted: u32 = capabilities.min_image_count + 1;
        let count: u32 = if capabilities.max_image_count > 0 {
            wanted.min(capabilities.max_image_count)
        } else {
            wanted
        };

        self.swapchain = unsafe {
            self.swapchain_api.create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .surface(self.surface)
                    .min_image_count(count)
                    .image_format(self.format)
                    .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                    .image_extent(self.extent)
                    .image_array_layers(1)
                    .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(capabilities.current_transform)
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    // FIFO is vsync, and the only mode every implementation must
                    // support. MAILBOX is the low-latency one where offered.
                    .present_mode(vk::PresentModeKHR::FIFO)
                    .clipped(true),
                None,
            )
        }
        .expect("create_swapchain");

        let images: Vec<vk::Image> =
            unsafe { self.swapchain_api.get_swapchain_images(self.swapchain) }
                .expect("get_swapchain_images");

        for image in &images {
            let view: vk::ImageView = unsafe {
                self.device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(*image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(self.format)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    None,
                )
            }
            .expect("create_image_view");

            let attachments = [view];
            let framebuffer: vk::Framebuffer = unsafe {
                self.device.create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(self.render_pass)
                        .attachments(&attachments)
                        .width(self.extent.width)
                        .height(self.extent.height)
                        .layers(1),
                    None,
                )
            }
            .expect("create_framebuffer");

            let finished: vk::Semaphore = unsafe {
                self.device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            }
            .expect("create_semaphore");

            self.views.push(view);
            self.framebuffers.push(framebuffer);
            self.render_finished.push(finished);
        }
    }

    /// Destroys everything that depends on the swapchain's size.
    fn destroy_swapchain(&mut self) {
        unsafe {
            for framebuffer in self.framebuffers.drain(..) {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            for view in self.views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
            for semaphore in self.render_finished.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }

            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_api.destroy_swapchain(self.swapchain, None);
                self.swapchain = vk::SwapchainKHR::null();
            }
        }
    }

    /// Gives everything back, in reverse order of creation.
    ///
    /// Called from `main` rather than from `Drop`, so that the order is visible
    /// rather than implied.
    pub fn destroy(&mut self) {
        unsafe {
            self.device.device_wait_idle().expect("device_wait_idle");

            self.destroy_swapchain();

            self.device.destroy_fence(self.in_flight, None);
            self.device.destroy_semaphore(self.image_available, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            self.device.destroy_device(None);
            self.surface_api.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

// ---------------------------------------------------------------------------
// Setup, split up so each step reads on its own
// ---------------------------------------------------------------------------

/// The instance and the surface to draw on.
fn create_instance_and_surface(
    entry: &Entry,
    window: &Window,
) -> (Instance, khr::surface::Instance, vk::SurfaceKHR) {
    let application = vk::ApplicationInfo::default()
        .application_name(c"voxel-world playground")
        .api_version(vk::API_VERSION_1_3);

    let display = window
        .display_handle()
        .expect("the window has a display handle");

    let mut extensions: Vec<*const i8> =
        ash_window::enumerate_required_extensions(display.as_raw())
            .expect("the platform's surface extensions")
            .to_vec();

    // MoltenVK is a portable implementation rather than a conformant one, so
    // the loader hides it unless asked to enumerate portable drivers.
    extensions.push(khr::portability_enumeration::NAME.as_ptr());
    extensions.push(khr::get_physical_device_properties2::NAME.as_ptr());

    let instance: Instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default()
                .application_info(&application)
                .enabled_extension_names(&extensions)
                .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR),
            None,
        )
    }
    .expect("create_instance");

    let surface: vk::SurfaceKHR = unsafe {
        ash_window::create_surface(
            entry,
            &instance,
            display.as_raw(),
            window
                .window_handle()
                .expect("the window has a window handle")
                .as_raw(),
            None,
        )
    }
    .expect("create_surface");

    let surface_api: khr::surface::Instance = khr::surface::Instance::new(entry, &instance);

    (instance, surface_api, surface)
}

/// The first device with a queue that can both draw and present here.
fn pick_device(
    instance: &Instance,
    surface_api: &khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> (vk::PhysicalDevice, u32) {
    let devices: Vec<vk::PhysicalDevice> =
        unsafe { instance.enumerate_physical_devices() }.expect("enumerate_physical_devices");

    for device in devices {
        let families = unsafe { instance.get_physical_device_queue_family_properties(device) };

        for (index, family) in families.iter().enumerate() {
            let index: u32 = index as u32;
            let draws: bool = family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let presents: bool =
                unsafe { surface_api.get_physical_device_surface_support(device, index, surface) }
                    .unwrap_or(false);

            if draws && presents {
                let properties = unsafe { instance.get_physical_device_properties(device) };
                let name = properties
                    .device_name_as_c_str()
                    .unwrap_or(c"unnamed")
                    .to_string_lossy()
                    .into_owned();
                println!("using {name}");

                return (device, index);
            }
        }
    }

    panic!("no device can both draw and present to this window");
}

/// The logical device and its one queue.
fn create_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queue_family: u32,
) -> (Device, vk::Queue) {
    let priorities = [1.0f32];
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&priorities)];

    let available: Vec<vk::ExtensionProperties> =
        unsafe { instance.enumerate_device_extension_properties(physical_device) }
            .expect("enumerate_device_extension_properties");

    let mut extensions: Vec<*const i8> = vec![khr::swapchain::NAME.as_ptr()];

    // Required by the spec whenever the implementation offers it, which on a
    // Mac it always does.
    let portable: bool = available
        .iter()
        .any(|extension| extension.extension_name_as_c_str() == Ok(khr::portability_subset::NAME));

    if portable {
        extensions.push(khr::portability_subset::NAME.as_ptr());
    }

    let device: Device = unsafe {
        instance.create_device(
            physical_device,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&extensions),
            None,
        )
    }
    .expect("create_device");

    let queue: vk::Queue = unsafe { device.get_device_queue(queue_family, 0) };

    (device, queue)
}

/// The format to draw in and the size to draw at.
fn surface_settings(
    surface_api: &khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    window: &Window,
) -> (vk::Format, vk::Extent2D) {
    let formats: Vec<vk::SurfaceFormatKHR> =
        unsafe { surface_api.get_physical_device_surface_formats(physical_device, surface) }
            .expect("get_physical_device_surface_formats");

    // sRGB where it is offered, so colours written in the shader land where
    // they look right; otherwise whatever comes first.
    let format: vk::Format = formats
        .iter()
        .find(|candidate| {
            candidate.format == vk::Format::B8G8R8A8_SRGB
                && candidate.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .or_else(|| formats.first())
        .expect("the surface supports at least one format")
        .format;

    let capabilities =
        unsafe { surface_api.get_physical_device_surface_capabilities(physical_device, surface) }
            .expect("get_physical_device_surface_capabilities");

    // A width of u32::MAX means "you choose", which is what a Mac reports.
    let extent: vk::Extent2D = if capabilities.current_extent.width == u32::MAX {
        let size = window.inner_size();

        vk::Extent2D {
            width: size.width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: size.height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    } else {
        capabilities.current_extent
    };

    (format, extent)
}

/// One subpass drawing to one colour attachment.
fn create_render_pass(device: &Device, format: vk::Format) -> vk::RenderPass {
    let attachments = [vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];

    let references = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&references)];

    // Keeps the subpass from writing before the image has been acquired.
    let dependencies = [vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

    unsafe {
        device.create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(&subpasses)
                .dependencies(&dependencies),
            None,
        )
    }
    .expect("create_render_pass")
}

/// The graphics pipeline: the two shaders and every fixed-function setting
/// between them.
fn create_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
) -> (vk::PipelineLayout, vk::Pipeline) {
    let vertex: vk::ShaderModule = load_shader(device, VERTEX_SPIRV);
    let fragment: vk::ShaderModule = load_shader(device, FRAGMENT_SPIRV);

    let entry_point: &CStr = c"main";
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex)
            .name(entry_point),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment)
            .name(entry_point),
    ];

    // No vertex buffer: the corners are constants in the shader.
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    // Set for real in the command buffer, so a resize needs no new pipeline.
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0);

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let blends = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(false)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blends);

    let pipeline_layout: vk::PipelineLayout =
        unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None) }
            .expect("create_pipeline_layout");

    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipeline: vk::Pipeline = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
    }
    .expect("create_graphics_pipelines")[0];

    // The modules are baked into the pipeline and are of no further use.
    unsafe {
        device.destroy_shader_module(vertex, None);
        device.destroy_shader_module(fragment, None);
    }

    (pipeline_layout, pipeline)
}

/// Turns compiled SPIR-V into a shader module.
fn load_shader(device: &Device, spirv: &[u8]) -> vk::ShaderModule {
    // SPIR-V is a stream of 32-bit words, and `include_bytes!` gives bytes that
    // are not necessarily aligned for one, so they are copied into a word
    // buffer rather than reinterpreted in place.
    let (words, remainder) = spirv.as_chunks::<4>();
    debug_assert!(remainder.is_empty(), "SPIR-V is a whole number of words");

    let words: Vec<u32> = words.iter().copied().map(u32::from_le_bytes).collect();

    unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
    }
    .expect("create_shader_module")
}

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
