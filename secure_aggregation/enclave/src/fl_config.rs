use std::collections::HashMap;
use std::collections::HashSet;
use std::vec::Vec;

#[derive(Clone, Default, Debug)]
pub struct FLConfigMap {
    pub configs: HashMap<u32, FLConfig>
}

impl FLConfigMap {
    pub fn new() -> Self {
        FLConfigMap::default()
    }

    pub fn add(& mut self, id: u32, config: FLConfig) {
        self.configs.insert(id, config);
    }
}

#[derive(Clone, Default, Debug)]
pub struct FLConfig {
    pub client_ids: Vec<u32>,
    pub client_size: usize,
    pub sigma: f32,
    pub clipping: f32,
    pub alpha: f32,
    pub sampling_ratio: f32,
    pub current_round: u32,
    pub aggregation_alg: u32,
    pub verbose: u8,
    pub dp: u8,
    pub current_sampled_clients: HashSet<u32>,
}

impl FLConfig {
    pub fn new() -> Self {
        FLConfig::default()
    }

    pub fn increment_round(&mut self) {
        self.current_round += 1
    }
}