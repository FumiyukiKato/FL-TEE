use std::collections::HashMap;
use std::collections::HashSet;
use std::vec::Vec;

type FLID = u32;

#[derive(Clone, Default, Debug)]
pub struct FLConfigMap {
    pub configs: HashMap<FLID, FLConfig>
}

impl FLConfigMap {
    pub fn new() -> Self {
        println!("[SGX] Build FLConfigMap Store");
        FLConfigMap::default()
    }

    pub fn add(& mut self, fl_id: u32, config: FLConfig) {
        self.configs.insert(fl_id, config);
    }
}

impl Drop for FLConfigMap {
    fn drop(&mut self) {
        println!("[SGX] (never called!!) FLConfigMap Dropped");
    }
}

#[derive(Clone, Default, Debug)]
pub struct FLConfig {
    pub client_ids: Vec<u32>,
    pub client_size: usize,
    pub num_of_parameters: usize,
    pub num_of_sparse_parameters: usize,
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

impl Drop for FLConfig {
    fn drop(&mut self) {
        println!("[SGX] FLConfig Dropped (correctly Overwritten without memory leak)");
    }
}