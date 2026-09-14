struct Octree {
    leaf: bool,
    children: Option<[Box<Octree>; 8]>,
}

