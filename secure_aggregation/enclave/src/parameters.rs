use std::vec::Vec;

pub type Index = u32;
const INDEX_BYTE_SIZE: usize = 4;
pub type Value = f32;
const VALUE_BYTE_SIZE: usize = 4;
pub const WEIGHT_BYTE_SIZE: usize = INDEX_BYTE_SIZE + VALUE_BYTE_SIZE;

#[derive(Clone, Default, Debug)]
pub struct Weight(pub Index, pub Value);

impl PartialEq for Weight {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for Weight {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

#[derive(Default, Debug)]
pub struct Parameters {
    pub weights: Vec<Weight>,
}

impl Parameters {
    pub fn new(num_of_parameters: usize) -> Self {
        let weights = Vec::with_capacity(num_of_parameters);
        Self { weights }
    }

    // pub fn global_init(num_of_parameters: usize) -> Vec<f32> {
    //     let mut weights = Vec::with_capacity(num_of_parameters);
    //     for _ in 0..(num_of_parameters as Index) {
    //         weights.push(0.0)
    //     }
    //     weights
    // }

    // pub fn initialize(num_of_parameters: usize) -> Self {
    //     let mut weights = Vec::with_capacity(num_of_parameters);
    //     for i in 0..(num_of_parameters as Index) {
    //         weights.push(Weight(i, 0.0))
    //     }
    //     Self { weights }
    // }

    pub fn make_weights_from_bytes(bytes: &Vec<u8>, num_of_parameters: usize) -> Vec<Weight> {
        let mut weights: Vec<Weight> = Vec::with_capacity(num_of_parameters);
        for i in 0..num_of_parameters {
            let start_pos = i * WEIGHT_BYTE_SIZE;
            let mut index_bytes: [u8; INDEX_BYTE_SIZE] = Default::default();
            let mut value_bytes: [u8; VALUE_BYTE_SIZE] = Default::default();
            index_bytes.copy_from_slice(&bytes[start_pos..start_pos + INDEX_BYTE_SIZE]);
            value_bytes
                .copy_from_slice(&bytes[start_pos + INDEX_BYTE_SIZE..start_pos + WEIGHT_BYTE_SIZE]);
            let index = Index::from_le_bytes(index_bytes);
            let value = Value::from_le_bytes(value_bytes);
            weights.push(Weight(index, value));
        }
        weights
    }
}
