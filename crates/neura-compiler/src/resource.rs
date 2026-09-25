use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

pub struct Read<T>(PhantomData<T>);
pub struct DynamicRead<T>(PhantomData<T>);
pub struct ReadWrite<T>(PhantomData<T>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Uvec3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl<T> Index<u32> for Read<T> {
    type Output = T;

    fn index(&self, _: u32) -> &Self::Output {
        unreachable!("a kernel resource exists only on the device")
    }
}

impl<T> Index<u32> for DynamicRead<T> {
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
