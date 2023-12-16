
use std::ffi::CString;

mod search;
use search::search;

mod utils;
use utils::load_onnx_model;

#[no_mangle]
pub fn mcts(model_path: *mut i8, n_games: i32, n_searches: i32, n_threads: i32) -> (i32, [f32; 7]) {

    // 1. Prepare input arguments
    let input_str: String = unsafe { CString::from_raw(model_path).into_string().unwrap() };
    let n_games: usize = n_games as usize;
    let n_searches: usize = n_searches as usize;
    let n_threads: usize = n_threads as usize;

    // 2. Create the pytorch model.
     let model = load_onnx_model(input_str);

    //let (winner, probs) = search(input_str, n_games, n_searches, n_threads);

    (0, [0.0; 7])
}