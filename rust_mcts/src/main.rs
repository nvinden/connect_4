extern crate rust_mcts;

use rust_mcts::mcts;
use std::ffi::CString;

fn main() {
    // Debugging mcts() function

    let model_path: &str = "models/model.pb";
    let c_model_path: CString = CString::new(model_path).unwrap();
    let model_path_ptr: *mut i8 = c_model_path.into_raw();

    let n_games: i32 = 1;
    let n_searches: i32 = 1;
    let n_threads: i32 = 1;

    let (winner, probs) = mcts(model_path_ptr, n_games, n_searches, n_threads);

    println!("winner: {}", winner);
    println!("probs: {:?}", probs);
}



