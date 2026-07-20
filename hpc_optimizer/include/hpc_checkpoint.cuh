// =============================================================================
// HPC CUDA Optimizer Library
// hpc_checkpoint.cuh  –  Binary checkpoint save / restore for optimizer state
//
// Saves and restores:
//   • Optimizer moment buffers (m, v, v_max) — always FP32
//   • Velocity buffers (SGD)
//   • Step counter
//   • Hyperparameter snapshot (config structs)
//   • Optional: FP32 master weight buffers
//
// Format (little-endian binary):
//   [8B magic] [4B version] [4B n_tensors] [4B step]
//   For each tensor:
//     [8B numel] [4B n_buffers]
//     For each buffer: [numel * 4B float data]
//   [sizeof(ConfigT) bytes – hyperparameters]
//
// HPC design:
//   • Uses cudaMemcpy D→H to pull moment state off device for serialisation.
//   • Rank-0 writes the checkpoint; all ranks read it back independently.
//   • Atomic rename (tmp → final) prevents corrupt checkpoints on crash.
//   • Supports partial restore: skips mismatched tensor counts with a warning.
// =============================================================================
#pragma once

#include "hpc_types.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

namespace hpc_opt {

static constexpr uint64_t HPC_CKPT_MAGIC   = 0x4F50545F48504300ULL; // "OPT_HPC\0"
static constexpr uint32_t HPC_CKPT_VERSION = 2;

// ===========================================================================
// CheckpointIO  –  generic save/restore for any collection of FP32 buffers
// ===========================================================================
class CheckpointIO {
public:
    // -----------------------------------------------------------------------
    // save
    //   path         – checkpoint file path (written atomically)
    //   d_buffers    – array of device buffer pointers (each FP32)
    //   numel_list   – number of elements in each buffer
    //   n_buffers    – total number of buffers (may span multiple tensors)
    //   n_tensors    – number of parameter tensors
    //   step         – current optimizer step counter
    //   config_blob  – raw bytes of hyperparameter struct
    //   config_size  – sizeof(config_blob)
    //   rank         – only rank-0 actually writes (others return immediately)
    // -----------------------------------------------------------------------
    static void save(const std::string& path,
                     float** const      d_buffers,
                     const size_t*      numel_list,
                     int                n_buffers,
                     int                n_tensors,
                     uint32_t           step,
                     const void*        config_blob,
                     size_t             config_size,
                     int                rank = 0)
    {
        if (rank != 0) return;

        std::string tmp = path + ".tmp";
        FILE* f = fopen(tmp.c_str(), "wb");
        if (!f) throw std::runtime_error("[CheckpointIO] cannot open " + tmp);

        // Header
        fwrite(&HPC_CKPT_MAGIC,   sizeof(HPC_CKPT_MAGIC),   1, f);
        fwrite(&HPC_CKPT_VERSION, sizeof(HPC_CKPT_VERSION), 1, f);

        uint32_t nb = static_cast<uint32_t>(n_buffers);
        uint32_t nt = static_cast<uint32_t>(n_tensors);
        fwrite(&nt,   sizeof(nt),   1, f);
        fwrite(&nb,   sizeof(nb),   1, f);
        fwrite(&step, sizeof(step), 1, f);

        // Buffer data
        for (int i = 0; i < n_buffers; ++i) {
            uint64_t numel = numel_list[i];
            fwrite(&numel, sizeof(numel), 1, f);

            if (numel > 0 && d_buffers[i]) {
                std::vector<float> h(numel);
                HPC_CUDA_CHECK(cudaMemcpy(h.data(), d_buffers[i],
                                          numel * sizeof(float),
                                          cudaMemcpyDeviceToHost));
                fwrite(h.data(), sizeof(float), numel, f);
            }
        }

        // Config blob
        uint64_t cs = config_size;
        fwrite(&cs, sizeof(cs), 1, f);
        if (cs > 0 && config_blob) fwrite(config_blob, 1, cs, f);

        fclose(f);

        // Atomic rename
        if (std::rename(tmp.c_str(), path.c_str()) != 0)
            throw std::runtime_error("[CheckpointIO] rename failed: " + tmp);

        fprintf(stdout, "[CheckpointIO] saved checkpoint: %s  (step=%u)\n",
                path.c_str(), step);
    }

    // -----------------------------------------------------------------------
    // load
    //   Returns step counter. Fills d_buffers from file.
    //   config_blob_out must point to pre-allocated buffer of config_size bytes.
    // -----------------------------------------------------------------------
    static uint32_t load(const std::string& path,
                         float**            d_buffers,
                         const size_t*      numel_list,
                         int                n_buffers,
                         void*              config_blob_out,
                         size_t             config_size,
                         cudaStream_t       stream = 0)
    {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("[CheckpointIO] cannot open " + path);

        // Verify header
        uint64_t magic;   fread(&magic,   sizeof(magic),   1, f);
        uint32_t version; fread(&version, sizeof(version), 1, f);

        if (magic != HPC_CKPT_MAGIC)
            throw std::runtime_error("[CheckpointIO] bad magic — not an hpc_opt checkpoint");
        if (version != HPC_CKPT_VERSION) {
            fprintf(stderr, "[CheckpointIO] version mismatch: file=%u expected=%u\n",
                    version, HPC_CKPT_VERSION);
        }

        uint32_t nt_file, nb_file, step;
        fread(&nt_file, sizeof(nt_file), 1, f);
        fread(&nb_file, sizeof(nb_file), 1, f);
        fread(&step,    sizeof(step),    1, f);

        int n_read = static_cast<int>(nb_file) < n_buffers
                     ? static_cast<int>(nb_file) : n_buffers;

        for (int i = 0; i < static_cast<int>(nb_file); ++i) {
            uint64_t numel_file;
            fread(&numel_file, sizeof(numel_file), 1, f);

            if (i >= n_buffers || numel_file == 0) {
                // Skip excess buffers
                fseek(f, static_cast<long>(numel_file * sizeof(float)), SEEK_CUR);
                continue;
            }

            if (numel_file != numel_list[i]) {
                fprintf(stderr, "[CheckpointIO] buffer %d: file numel=%llu != expected=%zu — skipping\n",
                        i, (unsigned long long)numel_file, numel_list[i]);
                fseek(f, static_cast<long>(numel_file * sizeof(float)), SEEK_CUR);
                continue;
            }

            std::vector<float> h(numel_file);
            fread(h.data(), sizeof(float), numel_file, f);

            if (d_buffers[i]) {
                HPC_CUDA_CHECK(cudaMemcpyAsync(d_buffers[i], h.data(),
                                               numel_file * sizeof(float),
                                               cudaMemcpyHostToDevice, stream));
            }
        }
        (void)n_read;

        // Config blob
        uint64_t cs; fread(&cs, sizeof(cs), 1, f);
        if (cs > 0 && config_blob_out) {
            size_t to_read = (cs < config_size) ? cs : config_size;
            fread(config_blob_out, 1, to_read, f);
        }

        fclose(f);
        fprintf(stdout, "[CheckpointIO] loaded checkpoint: %s  (step=%u)\n",
                path.c_str(), step);
        return step;
    }
};

// ===========================================================================
// AdamCheckpoint  –  convenience wrapper for AdamOptimizer state
// ===========================================================================
// Forward declaration — include hpc_adam.cuh before using this
template<typename AdamOpt>
inline void save_adam_checkpoint(
        const std::string& path,
        AdamOpt&           opt,
        const AdamConfig&  cfg,
        const size_t*      numel_list,
        int                rank = 0)
{
    int n = opt.n_tensors();
    std::vector<float*> bufs;
    std::vector<size_t> numels;

    // Interleave m[i], v[i] per tensor
    for (int i = 0; i < n; ++i) {
        bufs.push_back(opt.m_buffers()[i]);
        bufs.push_back(opt.v_buffers()[i]);
        numels.push_back(numel_list[i]);
        numels.push_back(numel_list[i]);
    }

    CheckpointIO::save(path, bufs.data(), numels.data(),
                       static_cast<int>(bufs.size()), n,
                       opt.step_count(),
                       &cfg, sizeof(cfg), rank);
}

template<typename AdamOpt>
inline uint32_t load_adam_checkpoint(
        const std::string& path,
        AdamOpt&           opt,
        AdamConfig&        cfg,
        const size_t*      numel_list,
        cudaStream_t       stream = 0)
{
    int n = opt.n_tensors();
    std::vector<float*> bufs;
    std::vector<size_t> numels;

    for (int i = 0; i < n; ++i) {
        bufs.push_back(opt.m_buffers()[i]);
        bufs.push_back(opt.v_buffers()[i]);
        numels.push_back(numel_list[i]);
        numels.push_back(numel_list[i]);
    }

    return CheckpointIO::load(path, bufs.data(), numels.data(),
                              static_cast<int>(bufs.size()),
                              &cfg, sizeof(cfg), stream);
}

} // namespace hpc_opt
