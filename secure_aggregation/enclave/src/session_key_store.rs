use std::collections::HashMap;
use std::vec::Vec;

type ClientId = u32;
type SessionKey = [u8; 16];

#[derive(Clone, Default, Debug)]
pub struct SessionKeyStore {
    pub map: HashMap<u32, SessionKey>
}

impl SessionKeyStore {
    pub fn new() -> Self {
        SessionKeyStore::default()
    }

    pub fn build_mock(client_ids: Vec<u32>) -> Self {
        println!("[SGX] Build Remote Attestation mock session keys");
        let mut map: HashMap<ClientId, SessionKey> = HashMap::with_capacity(client_ids.len());
        for client_id in client_ids.iter() {
            let mut shared_key: [u8; 16] = [0; 16];
            shared_key[4..8].copy_from_slice(&client_id.to_be_bytes());
            map.insert(*client_id, shared_key);
        }
        Self { map }
    }
}

impl Drop for SessionKeyStore {
    fn drop(&mut self) {
        println!("[SGX] (never called!!) SessionKeyStore Dropped");
    }
}