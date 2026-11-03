use libc::c_int;

pub static kCLDFlagScoreAsQuads: c_int = 0x0100;
pub static kCLDFlagHtml: c_int =         0x0200;
pub static kCLDFlagCr: c_int =           0x0400;
pub static kCLDFlagVerbose: c_int =      0x0800;
pub static kCLDFlagQuiet: c_int =        0x1000;
pub static kCLDFlagEcho: c_int =         0x2000;

