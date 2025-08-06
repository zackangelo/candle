use cudarc::{
    cublas::result::CublasError,
    cublaslt::{safe::*, sys},
    driver::DevicePtr,
};

/// Matrix matrix multiplication with elements of type `T`.
pub trait Matmul<T>: MatmulShared {
    /// Underlying CUDA Type for `T`
    fn matrix_type() -> sys::cudaDataType;

    /// Underlying CUDA Compute Type for `T`
    fn compute_type() -> sys::cublasComputeType_t;

    /// Matrix matrix multiplication. See
    /// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul)
    ///
    /// # Safety
    /// This is unsafe because improper arguments may lead to invalid
    /// memory accesses.
    unsafe fn matmul<I: DevicePtr<T>, O: DevicePtrMut<T>>(
        &self,
        cfg: MatmulConfig,
        a: &I,
        b: &I,
        c: &mut O,
        bias: Option<&I>,
        act: Option<&Activation>,
    ) -> Result<(), CublasError> {
        let (a_rows, a_cols) = if cfg.transa {
            (cfg.k, cfg.m)
        } else {
            (cfg.m, cfg.k)
        };
        let (b_rows, b_cols) = if cfg.transb {
            (cfg.n, cfg.k)
        } else {
            (cfg.k, cfg.n)
        };

        // Creates matrix layouts
        let a_layout = MatrixLayout::new(Self::matrix_type(), a_rows, a_cols, cfg.lda)?;
        if let (Some(batch_size), Some(stride_a)) = (cfg.batch_size, cfg.stride_a) {
            a_layout.set_batch(batch_size, stride_a)?;
        }

        let b_layout = MatrixLayout::new(Self::matrix_type(), b_rows, b_cols, cfg.ldb)?;
        if let (Some(batch_size), Some(stride_b)) = (cfg.batch_size, cfg.stride_b) {
            b_layout.set_batch(batch_size, stride_b)?;
        }

        let c_layout = MatrixLayout::new(Self::matrix_type(), cfg.m, cfg.n, cfg.ldc)?;
        if let (Some(batch_size), Some(stride_c)) = (cfg.batch_size, cfg.stride_c) {
            c_layout.set_batch(batch_size, stride_c)?;
        }

        // Matmul description
        let matmul_desc = MatmulDesc::new(Self::compute_type(), sys::cudaDataType_t::CUDA_R_32F)?;

        // Set transa
        matmul_desc.set_transpose(cfg.transa, Matrix::A)?;
        // Set transb
        matmul_desc.set_transpose(cfg.transb, Matrix::B)?;

        // Epilogue system can be leveraged to fuse add and activation operations
        matmul_desc.set_epilogue(act, bias.map(|b| b.device_ptr()), cfg.stride_bias)?;

        // Create matmul heuristic search preferences
        let matmul_pref = MatmulPref::new()?;

        // Set workspace size
        matmul_pref.set_workspace_size(self.workspace().size)?;

        // Get heuristic given Config, bias, act and workspace size
        let heuristic = result::get_matmul_algo_heuristic(
            *self.handle(),
            matmul_desc.handle,
            a_layout.handle,
            b_layout.handle,
            c_layout.handle,
            c_layout.handle,
            matmul_pref.handle,
        )?;

        // Launch matmul kernel
        result::matmul(
            *self.handle(),
            matmul_desc.handle,
            (&cfg.alpha) as *const _ as *const _,
            (&cfg.beta) as *const _ as *const _,
            *a.device_ptr() as *const _,
            a_layout.handle,
            *b.device_ptr() as *const _,
            b_layout.handle,
            *c.device_ptr_mut() as *const _,
            c_layout.handle,
            *c.device_ptr_mut() as *mut _,
            c_layout.handle,
            (&heuristic.algo) as *const _,
            *self.workspace().buffer.device_ptr() as *const CUdeviceptr as *mut _,
            self.workspace().size,
            *self.stream() as *mut _,
        )
    }
}
