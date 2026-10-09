use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

pub struct Read<T>(PhantomData<T>);
pub struct ReadWrite<T>(PhantomData<T>);

pub struct Workgroup<T, const N: usize>(PhantomData<T>);

impl<T, const N: usize> Workgroup<T, N> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T, const N: usize> Default for Workgroup<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Index<u32> for Workgroup<T, N> {
    type Output = T;

    fn index(&self, _: u32) -> &Self::Output {
        unreachable!("device workgroup memory exists only on the device")
    }
}

impl<T, const N: usize> IndexMut<u32> for Workgroup<T, N> {
    fn index_mut(&mut self, _: u32) -> &mut Self::Output {
        unreachable!("device workgroup memory exists only on the device")
    }
}

pub struct AtomicU32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Uvec3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Uvec4 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub w: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fvec2 {
    pub x: f32,
    pub y: f32,
}

impl<T> Index<u32> for Read<T> {
    type Output = T;

    fn index(&self, _: u32) -> &Self::Output {
        unreachable!("a kernel resource exists only on the device")
    }
}

impl<T> Index<u32> for ReadWrite<T> {
    type Output = T;

    fn index(&self, _: u32) -> &Self::Output {
        unreachable!("a kernel resource exists only on the device")
    }
}

impl<T> IndexMut<u32> for ReadWrite<T> {
    fn index_mut(&mut self, _: u32) -> &mut Self::Output {
        unreachable!("a kernel resource exists only on the device")
    }
}
