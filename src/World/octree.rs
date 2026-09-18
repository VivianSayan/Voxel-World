use crate::misc::random::Random;

enum Octype{
    Children {
        children: Option<[Box<Octree>; 8]>,
    },
    Uniform {
        id: usize
    },
    Probability {
        probabilities: Vec<f64>,
        ids: Vec<usize>,
    },
    Provedural, // Add so it can support noises must be called to check for certain things here when more detailed ones are collapsed.
}

struct Octree {
    octype: Octype,
}


impl Octree {
    pub fn new_uniform(id: usize) -> Self {
        Octree {
            octype: Octype::Uniform { id },
        }
    }

    pub fn new_children() -> Self {
        Octree {
            octype: Octype::Children { children: None },
        }
    }

    pub fn new_probability(probabilities: Vec<f64>, ids: Vec<usize>) -> Self {
        debug_assert!(probabilities.len() == ids.len(), "Probabilities and IDs must have the same length");
        let total_probability: f64 = probabilities.iter().sum();
        debug_assert!(total_probability > 0.0, "Total probability must be greater than zero");
        let mut new_probabilities: Vec<f64> = Vec::new();
        for prob in &probabilities {
            new_probabilities.push(prob + new_probabilities.last().unwrap_or(&0.0));
        }
        Octree {
            octype: Octype::Probability { probabilities: new_probabilities, ids },
        }
    }
}