#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include <libhmm/common/common.h>

namespace nb = nanobind;
using namespace libhmm;

using NpArray1DIn = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using NpArray2DIn = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

inline nb::object buf_to_numpy_owned(double *buf, size_t n) {
    nb::capsule owner(buf, [](void *p) noexcept { delete[] static_cast<double *>(p); });
    return nb::cast(
        nb::ndarray<nb::numpy, double, nb::ndim<1>>(buf, {n}, owner),
        nb::rv_policy::move);
}

inline nb::object buf_to_numpy_owned(std::int64_t *buf, size_t n) {
    nb::capsule owner(buf, [](void *p) noexcept { delete[] static_cast<std::int64_t *>(p); });
    return nb::cast(
        nb::ndarray<nb::numpy, std::int64_t, nb::ndim<1>>(buf, {n}, owner),
        nb::rv_policy::move);
}

inline ObservationSet observation_set_from_numpy(NpArray1DIn values) {
    ObservationSet out(values.shape(0));
    const double *data = values.data();
    for (size_t i = 0; i < values.shape(0); ++i) {
        out(i) = data[i];
    }
    return out;
}

inline Vector vector_from_numpy(NpArray1DIn values) {
    Vector out(values.shape(0));
    const double *data = values.data();
    for (size_t i = 0; i < values.shape(0); ++i) {
        out(i) = data[i];
    }
    return out;
}

inline Matrix matrix_from_numpy(NpArray2DIn values) {
    const size_t rows = values.shape(0);
    const size_t cols = values.shape(1);
    Matrix out(rows, cols);
    const double *data = values.data();
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            out(r, c) = data[r * cols + c];
        }
    }
    return out;
}

inline ObservationLists observation_lists_from_python(const nb::list &sequences) {
    ObservationLists out;
    out.reserve(nb::len(sequences));
    for (nb::handle item : sequences) {
        auto arr = nb::cast<NpArray1DIn>(item);
        out.push_back(observation_set_from_numpy(arr));
    }
    return out;
}

inline nb::object vector_to_numpy(const Vector &v) {
    const size_t n = v.size();
    auto *buf = new double[n];
    for (size_t i = 0; i < n; ++i) {
        buf[i] = v(i);
    }
    return buf_to_numpy_owned(buf, n);
}

inline nb::object observation_set_to_numpy(const ObservationSet &v) {
    const size_t n = v.size();
    auto *buf = new double[n];
    for (size_t i = 0; i < n; ++i) {
        buf[i] = v(i);
    }
    return buf_to_numpy_owned(buf, n);
}

inline nb::object state_sequence_to_numpy(const StateSequence &v) {
    const size_t n = v.size();
    auto *buf = new std::int64_t[n];
    for (size_t i = 0; i < n; ++i) {
        buf[i] = static_cast<std::int64_t>(v(i));
    }
    return buf_to_numpy_owned(buf, n);
}

inline nb::object matrix_to_numpy(const Matrix &m) {
    const size_t rows = m.size1();
    const size_t cols = m.size2();
    auto *buf = new double[rows * cols];
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            buf[r * cols + c] = m(r, c);
        }
    }
    nb::capsule owner(buf, [](void *p) noexcept { delete[] static_cast<double *>(p); });
    return nb::cast(
        nb::ndarray<nb::numpy, double, nb::ndim<2>>(buf, {rows, cols}, owner),
        nb::rv_policy::move);
}

template <typename Dist>
nb::object batch_log_pdf(const Dist &dist, NpArray1DIn x) {
    const size_t n = x.shape(0);
    auto *buf = new double[n];
    {
        nb::gil_scoped_release release;
        dist.getBatchLogProbabilities(
            std::span<const double>{x.data(), n},
            std::span<double>{buf, n});
    }
    return buf_to_numpy_owned(buf, n);
}
