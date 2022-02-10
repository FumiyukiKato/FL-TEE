#[inline]
pub fn o_equal(x: usize, y: usize) -> bool {
    let ret: bool;
    unsafe {
        llvm_asm!(
            "cmp %rcx, %rdx \n\t
             sete %al \n\t"
            : "={al}"(ret)
            : "{rcx}"(x), "{rdx}" (y)
            : "rcx", "rdx"
            : "volatile"
        );
    }
    ret
}

// set byte if below
#[inline]
pub fn o_setb(x: usize, y: usize) -> bool {
    let ret: bool;
    unsafe {
        llvm_asm!(
            "cmp %rdx, %rcx \n\t
             setb %al \n\t"
            : "={al}"(ret)
            : "{rcx}"(x), "{rdx}" (y)
            : "rcx", "rdx"
            : "volatile"
        );
    }
    ret
}

// Note: oswap swaps 8byte once.
// it means that parameters::Weight (= 4byte Index + 4byte Value) is swapped together.
// (The memory allocation of Weight depends on compiler? Maybe not)
// https://hal.inria.fr/hal-01512970v1/document
#[inline]
pub fn o_swap<T>(flag: isize, x: &T, y: &T) {
    unsafe {
        llvm_asm!(
            "test %rax, %rax \n\t
             movq (%r8), %r10 \n\t
             movq (%rdx), %r9 \n\t
             mov %r9, %r11 \n\t
             cmovnz %r10, %r9 \n\t
             cmovnz %r11, %r10 \n\t
             movq %r9, (%rdx) \n\t
             movq %r10, (%r8) \n\t"
            :
            : "{rax}"(flag), "{rdx}" (x), "{r8}" (y)
            : "rax", "rdx", "r8", "r9", "r10", "r11"
            : "volatile"
        );
    }
}


// T is 8byte at most.
#[inline]
pub fn o_mov<T>(flag: isize, src: T, val: T) -> T {
    let ret: T;
    unsafe {
        llvm_asm!(
            "xor %rcx, %rcx \n\t
            mov %r8, %rcx \n\t
            test %rcx, %rcx \n\t
            cmovnz %rdx, %rax \n\t"
            : "={rax}"(ret)
            : "{r8}"(flag), "{rax}" (src), "{rdx}" (val)
            : "rax", "rcx", "rdx", "r8"
            : "volatile"
        );
    }
    ret
}
