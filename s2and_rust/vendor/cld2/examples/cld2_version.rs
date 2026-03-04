//! This can be used to check the version of `cld2` we're using.

extern crate cld2;

use cld2::detector_version;

fn main() {
    println!("cld2 version: {}", detector_version());
}
