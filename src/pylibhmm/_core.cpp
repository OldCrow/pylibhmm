/// pylibhmm native extension.
///
/// Binds core libhmm model, calculator, trainer, and distribution APIs to
/// Python via nanobind.  The module is organised as five focused binder
/// functions (bind_distributions, bind_hmm, bind_calculators, bind_trainers,
/// bind_io) called from NB_MODULE.
///
/// Design notes
/// ============
///
/// Array I/O
///   All NumPy ⇔ libhmm conversions go through _common.h helpers.  Buffers
///   are copied into libhmm value types (Vector, Matrix, ObservationSet);
///   no raw pointer aliasing between Python and C++ storage.
///
/// Distribution cloning
///   Hmm::set_distribution takes ownership.  clone_distribution dispatches
///   through clone_registry (std::type_index → copy-ctor wrapper) for O(1)
///   lookup, avoiding a linear dynamic_cast chain.
///
/// GIL handling
///   Calls to train(), compute(), and decode() release the Python GIL via
///   nb::gil_scoped_release so other Python threads can run concurrently.
///
/// Lifetime management
///   Calculator and trainer constructors carry nb::keep_alive<1,2>() so
///   nanobind keeps the HMM alive at least as long as the dependent object.
///   get_distribution() returns a borrowed raw pointer
///   (nb::rv_policy::reference_internal) into the HMM's internal storage;
///   it must not outlive the Hmm.
///
/// Trainer holders
///   libhmm v4 BasicTrainer stores observations by const reference_wrapper,
///   not by value (v4 breaking change).  Python __init__ lambdas would create
///   local ObservationLists that die when the lambda returns, leaving every
///   trainer with a dangling obsLists_ref_.  Each trainer is wrapped in a
///   *Holder struct that owns the data on the heap (unique_ptr — stable
///   address) and passes *data_ to the inner trainer.  data_ is declared
///   before trainer_ so C++ member-init order guarantees it is alive when
///   trainer_ stores std::cref(*data_).
///
/// Compiler portability
///   PYLIBHMM_HAS_REQUIRES_EXPR selects C++20 requires expressions (preferred)
///   vs. C++17 std::void_t detection traits (fallback for Apple Clang 12 /
///   Catalina, which reports __cpp_concepts=201907L and rejects requires in
///   if constexpr conditions).

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <random>
#include <span>
#include <type_traits>
#include <typeindex>
#include <unordered_map>

#include <libhmm/calculators/basic_forward_backward_calculator.h>
#include <libhmm/calculators/forward_backward_calculator.h>
#include <libhmm/calculators/viterbi_calculator.h>
#include <libhmm/distributions/beta_distribution.h>
#include <libhmm/distributions/binomial_distribution.h>
#include <libhmm/distributions/chi_squared_distribution.h>
#include <libhmm/distributions/discrete_distribution.h>
#include <libhmm/distributions/diagonal_gaussian_distribution.h>
#include <libhmm/distributions/emission_distribution.h>
#include <libhmm/distributions/full_covariance_gaussian_distribution.h>
#include <libhmm/distributions/independent_components_distribution.h>
#include <libhmm/distributions/exponential_distribution.h>
#include <libhmm/distributions/gamma_distribution.h>
#include <libhmm/distributions/gaussian_distribution.h>
#include <libhmm/distributions/log_normal_distribution.h>
#include <libhmm/distributions/negative_binomial_distribution.h>
#include <libhmm/distributions/pareto_distribution.h>
#include <libhmm/distributions/poisson_distribution.h>
#include <libhmm/distributions/rayleigh_distribution.h>
#include <libhmm/distributions/student_t_distribution.h>
#include <libhmm/distributions/uniform_distribution.h>
#include <libhmm/distributions/von_mises_distribution.h>
#include <libhmm/distributions/weibull_distribution.h>
#include <libhmm/hmm.h>
#include <libhmm/io/hmm_json.h>
#include <libhmm/io/xml_file_reader.h>
#include <libhmm/io/xml_file_writer.h>
#include <libhmm/training/basic_baum_welch_trainer.h>
#include <libhmm/training/baum_welch_trainer.h>
#include <libhmm/training/kmeans_init.h>
#include <libhmm/training/map_baum_welch_trainer.h>
#include <libhmm/training/model_selection.h>
#include <libhmm/training/segmental_kmeans_trainer.h>
#include <libhmm/training/viterbi_trainer.h>

// =============================================================================
// Suppress implicit instantiation of all specialisations that have
// explicit instantiations in libhmm.a (compiled there with specific flags).
// Without these, _core.cpp generates conflicting implicit instantiations
// that cause ODR violations -> vtable mismatches -> segfaults on train().
// =============================================================================
extern template class libhmm::BasicForwardBackwardCalculator<double>;
extern template class libhmm::BasicForwardBackwardCalculator<libhmm::ObservationVectorView>;
extern template class libhmm::BasicViterbiCalculator<double>;
extern template class libhmm::BasicViterbiCalculator<libhmm::ObservationVectorView>;
extern template class libhmm::BasicBaumWelchTrainer<double>;
extern template class libhmm::BasicBaumWelchTrainer<libhmm::ObservationVectorView>;
extern template class libhmm::BasicMapBaumWelchTrainer<double>;
extern template class libhmm::BasicMapBaumWelchTrainer<libhmm::ObservationVectorView>;
extern template class libhmm::BasicViterbiTrainer<double>;
extern template class libhmm::BasicViterbiTrainer<libhmm::ObservationVectorView>;
extern template class libhmm::BasicSegmentalKMeansTrainer<double>;
extern template class libhmm::BasicSegmentalKMeansTrainer<libhmm::ObservationVectorView>;

#include "_common.h"

namespace nb = nanobind;
using namespace libhmm;

namespace {

// ---------------------------------------------------------------------------
// Capability detection for optional distribution methods.
//
// Primary path: C++20 requires expressions in if constexpr conditions.
// Fallback path: C++17 std::void_t detection idiom, used on compilers that
// advertise partial C++20 concepts support but do not accept requires
// expressions as constexpr conditions (e.g. Apple Clang 12, Xcode 12,
// macOS Catalina 10.15 — reports __cpp_concepts=201907L).
// ---------------------------------------------------------------------------
#if defined(__cpp_concepts) && __cpp_concepts >= 202002L
#  define PYLIBHMM_HAS_REQUIRES_EXPR 1
#else
#  define PYLIBHMM_HAS_REQUIRES_EXPR 0

template <typename T, typename = void>
struct has_get_cumulative_probability : std::false_type {};
template <typename T>
struct has_get_cumulative_probability<T,
    std::void_t<decltype(std::declval<const T &>().getCumulativeProbability(0.0))>>
    : std::true_type {};

template <typename T, typename = void>
struct has_CDF : std::false_type {};
template <typename T>
struct has_CDF<T, std::void_t<decltype(std::declval<const T &>().CDF(0.0))>>
    : std::true_type {};

template <typename T, typename = void>
struct has_get_mean : std::false_type {};
template <typename T>
struct has_get_mean<T, std::void_t<decltype(std::declval<const T &>().getMean())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_get_variance : std::false_type {};
template <typename T>
struct has_get_variance<T, std::void_t<decltype(std::declval<const T &>().getVariance())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_get_standard_deviation : std::false_type {};
template <typename T>
struct has_get_standard_deviation<T,
    std::void_t<decltype(std::declval<const T &>().getStandardDeviation())>>
    : std::true_type {};

#endif // !PYLIBHMM_HAS_REQUIRES_EXPR

/// Thread-local RNG for no-arg sample() calls. Each thread initialises its
/// own instance from std::random_device on first access; safe for concurrent
/// no-arg sample() calls without locking.
static thread_local std::mt19937_64 g_rng{std::random_device{}()}; // NOLINT(cert-msc51-cpp)

template <typename Dist, typename PyClass>
void bind_distribution_common(PyClass &cls) {
    cls.def("pdf",
            [](const Dist &d, double x) { return d.getProbability(x); },
            nb::arg("x"),
            "Evaluate scalar probability density/mass at x.")
        .def("log_pdf",
             [](const Dist &d, double x) { return d.getLogProbability(x); },
             nb::arg("x"),
             "Evaluate scalar log probability density/mass at x.")
        .def("log_pdf",
             [](const Dist &d, NpArray1DIn x) -> nb::object { return batch_log_pdf(d, x); },
             nb::arg("x").noconvert(),
             "Evaluate batched log probability density/mass for a 1-D float64 NumPy array.")
        .def("fit",
             [](Dist &d, NpArray1DIn data) {
                 d.fit(std::span<const double>{data.data(), data.shape(0)});
             },
             nb::arg("data").noconvert(),
             "Fit distribution parameters to unweighted data.")
        .def("fit_weighted",
             [](Dist &d, NpArray1DIn data, NpArray1DIn weights) {
                 if (data.shape(0) != weights.shape(0)) {
                     throw nb::value_error("data and weights must have identical lengths");
                 }
                 d.fit(std::span<const double>{data.data(), data.shape(0)},
                       std::span<const double>{weights.data(), weights.shape(0)});
             },
             nb::arg("data").noconvert(),
             nb::arg("weights").noconvert(),
             "Fit distribution parameters to weighted data.")
        .def("sample",
             [](const Dist &d) -> double { return d.sample(g_rng); },
             "Draw one sample (non-deterministic; uses the module-level RNG).")
        .def("sample",
             [](const Dist &d, uint64_t seed) -> double {
                 std::mt19937_64 rng{seed};
                 return d.sample(rng);
             },
             nb::arg("seed"),
             "Draw one sample using the given integer seed (reproducible).")
        .def("reset", &Dist::reset, "Reset distribution parameters to defaults.")
        .def_prop_ro("is_discrete", &Dist::isDiscrete)
        // repr(d) returns the distribution's text representation (toString()).
        // Use named properties (d.mu, d.sigma, …) for programmatic parameter access.
        .def("__repr__", [](const Dist &d) { return d.toString(); });

#if PYLIBHMM_HAS_REQUIRES_EXPR
    if constexpr (requires(const Dist &d, double x) { d.getCumulativeProbability(x); }) {
        cls.def("cdf",
                [](const Dist &d, double x) { return d.getCumulativeProbability(x); },
                nb::arg("x"),
                "Evaluate scalar cumulative distribution function.");
    } else if constexpr (requires(const Dist &d, double x) { d.CDF(x); }) {
        cls.def("cdf",
                [](const Dist &d, double x) { return d.CDF(x); },
                nb::arg("x"),
                "Evaluate scalar cumulative distribution function.");
    }
    if constexpr (requires(const Dist &d) { d.getMean(); }) {
        cls.def_prop_ro("mean", &Dist::getMean);
    }
    if constexpr (requires(const Dist &d) { d.getVariance(); }) {
        cls.def_prop_ro("variance", &Dist::getVariance);
    }
    if constexpr (requires(const Dist &d) { d.getStandardDeviation(); }) {
        cls.def_prop_ro("std", &Dist::getStandardDeviation);
    }
#else  // void_t fallback for Apple Clang 12 / Catalina
    if constexpr (has_get_cumulative_probability<Dist>::value) {
        cls.def("cdf",
                [](const Dist &d, double x) { return d.getCumulativeProbability(x); },
                nb::arg("x"),
                "Evaluate scalar cumulative distribution function.");
    } else if constexpr (has_CDF<Dist>::value) {
        cls.def("cdf",
                [](const Dist &d, double x) { return d.CDF(x); },
                nb::arg("x"),
                "Evaluate scalar cumulative distribution function.");
    }
    if constexpr (has_get_mean<Dist>::value) {
        cls.def_prop_ro("mean", &Dist::getMean);
    }
    if constexpr (has_get_variance<Dist>::value) {
        cls.def_prop_ro("variance", &Dist::getVariance);
    }
    if constexpr (has_get_standard_deviation<Dist>::value) {
        cls.def_prop_ro("std", &Dist::getStandardDeviation);
    }
#endif // PYLIBHMM_HAS_REQUIRES_EXPR
}

// ---------------------------------------------------------------------------
// Distribution clone registry — maps std::type_index to a copy-constructor
// wrapper. Eliminates the linear dynamic_cast chain for set_distribution.
// ---------------------------------------------------------------------------
using CloneFn = std::function<std::unique_ptr<EmissionDistribution>(const EmissionDistribution &)>;

template <typename Dist>
CloneFn make_clone() {
    return [](const EmissionDistribution &base) -> std::unique_ptr<EmissionDistribution> {
        return std::make_unique<Dist>(static_cast<const Dist &>(base));
    };
}

/// Returns the global clone registry, initialised once on first call.
/// The static-local initialisation is thread-safe per C++11 §6.7.
const std::unordered_map<std::type_index, CloneFn> &clone_registry() {
    static const std::unordered_map<std::type_index, CloneFn> reg{
        {typeid(BetaDistribution),             make_clone<BetaDistribution>()},
        {typeid(BinomialDistribution),         make_clone<BinomialDistribution>()},
        {typeid(ChiSquaredDistribution),       make_clone<ChiSquaredDistribution>()},
        {typeid(DiscreteDistribution),         make_clone<DiscreteDistribution>()},
        {typeid(ExponentialDistribution),      make_clone<ExponentialDistribution>()},
        {typeid(GammaDistribution),            make_clone<GammaDistribution>()},
        {typeid(GaussianDistribution),         make_clone<GaussianDistribution>()},
        {typeid(LogNormalDistribution),        make_clone<LogNormalDistribution>()},
        {typeid(NegativeBinomialDistribution), make_clone<NegativeBinomialDistribution>()},
        {typeid(ParetoDistribution),           make_clone<ParetoDistribution>()},
        {typeid(PoissonDistribution),          make_clone<PoissonDistribution>()},
        {typeid(RayleighDistribution),         make_clone<RayleighDistribution>()},
        {typeid(StudentTDistribution),         make_clone<StudentTDistribution>()},
        {typeid(UniformDistribution),          make_clone<UniformDistribution>()},
        {typeid(VonMisesDistribution),         make_clone<VonMisesDistribution>()},
        {typeid(WeibullDistribution),          make_clone<WeibullDistribution>()},
    };
    return reg;
}

/// Returns a heap-allocated copy of @p distribution.
/// @throws nb::type_error if the concrete type is not registered.
[[nodiscard]] std::unique_ptr<EmissionDistribution>
clone_distribution(const EmissionDistribution &distribution) {
    const auto it = clone_registry().find(typeid(distribution));
    if (it == clone_registry().end()) {
        throw nb::type_error("Unsupported EmissionDistribution subtype");
    }
    return it->second(distribution);
}

// ---------------------------------------------------------------------------
// Trainer holders — own observation data on the heap so the trainer's
// reference_wrapper<const ListType> always points to live storage.
//
// Member declaration order matters: data_ before trainer_ so that data_ is
// fully constructed before trainer_(hmm, *data_) runs.
// ---------------------------------------------------------------------------

struct BaumWelchHolder {
    std::unique_ptr<ObservationLists> data_;
    BaumWelchTrainer                  trainer_;
    BaumWelchHolder(Hmm& hmm, ObservationLists obs)
        : data_(std::make_unique<ObservationLists>(std::move(obs)))
        , trainer_(hmm, *data_) {}
};

struct MapBaumWelchHolder {
    std::unique_ptr<ObservationLists> data_;
    MapBaumWelchTrainer               trainer_;
    MapBaumWelchHolder(Hmm& hmm, ObservationLists obs, double pc = 1.0)
        : data_(std::make_unique<ObservationLists>(std::move(obs)))
        , trainer_(hmm, *data_, pc) {}
};

struct ViterbiHolder {
    std::unique_ptr<ObservationLists> data_;
    ViterbiTrainer                    trainer_;
    ViterbiHolder(Hmm& hmm, ObservationLists obs, TrainingConfig cfg = {})
        : data_(std::make_unique<ObservationLists>(std::move(obs)))
        , trainer_(hmm, *data_, cfg) {}
};

struct SegmentalKMeansHolder {
    std::unique_ptr<ObservationLists> data_;
    SegmentalKMeansTrainer             trainer_;
    SegmentalKMeansHolder(Hmm& hmm, ObservationLists obs, std::size_t max_iters = 100)
        : data_(std::make_unique<ObservationLists>(std::move(obs)))
        , trainer_(hmm, *data_, max_iters) {}
};

using MVSKMTrainer = BasicSegmentalKMeansTrainer<ObservationVectorView>;
struct MVSegmentalKMeansHolder {
    std::unique_ptr<MultiObservationLists> data_;
    MVSKMTrainer                           trainer_;
    MVSegmentalKMeansHolder(HmmMV& hmm, MultiObservationLists obs, std::size_t max_iters = 100)
        : data_(std::make_unique<MultiObservationLists>(std::move(obs)))
        , trainer_(hmm, *data_, max_iters) {}
};

using MVBWT = BasicBaumWelchTrainer<ObservationVectorView>;
struct MVBaumWelchHolder {
    std::unique_ptr<MultiObservationLists> data_;
    MVBWT                                  trainer_;
    MVBaumWelchHolder(HmmMV& hmm, MultiObservationLists obs)
        : data_(std::make_unique<MultiObservationLists>(std::move(obs)))
        , trainer_(hmm, *data_) {}
};

using MVMAPBWTrainer = BasicMapBaumWelchTrainer<ObservationVectorView>;
struct MVMapBaumWelchHolder {
    std::unique_ptr<MultiObservationLists> data_;
    MVMAPBWTrainer                         trainer_;
    MVMapBaumWelchHolder(HmmMV& hmm, MultiObservationLists obs, double pc = 1.0)
        : data_(std::make_unique<MultiObservationLists>(std::move(obs)))
        , trainer_(hmm, *data_, pc) {}
};

// ---------------------------------------------------------------------------
// bind_distributions — base EmissionDistribution and all concrete subtypes.
// ---------------------------------------------------------------------------
void bind_distributions(nb::module_ &m) {
    auto emission = nb::class_<EmissionDistribution>(m, "EmissionDistribution");
    emission.def("pdf", &EmissionDistribution::getProbability, nb::arg("x"))
        .def("log_pdf", &EmissionDistribution::getLogProbability, nb::arg("x"))
        .def("sample",
             [](const EmissionDistribution &d) -> double { return d.sample(g_rng); },
             "Draw one sample (non-deterministic; uses the module-level RNG).")
        .def("sample",
             [](const EmissionDistribution &d, uint64_t seed) -> double {
                 std::mt19937_64 rng{seed};
                 return d.sample(rng);
             },
             nb::arg("seed"),
             "Draw one sample using the given integer seed (reproducible).")
        .def("reset", &EmissionDistribution::reset)
        .def_prop_ro("is_discrete", &EmissionDistribution::isDiscrete)
        // repr(d) returns the distribution's text representation (toString()).
        .def("__repr__", [](const EmissionDistribution &d) { return d.toString(); });

    auto discrete = nb::class_<DiscreteDistribution, EmissionDistribution>(
        m, "Discrete", "Discrete categorical distribution.");
    discrete.def(nb::init<int>(), nb::arg("num_symbols") = 10)
        .def("set_probability", &DiscreteDistribution::setProbability, nb::arg("index"), nb::arg("value"))
        .def("get_symbol_probability", &DiscreteDistribution::getSymbolProbability, nb::arg("index"))
        .def_prop_ro("num_symbols", &DiscreteDistribution::getNumSymbols)
        .def_prop_ro("mode", &DiscreteDistribution::getMode);
    bind_distribution_common<DiscreteDistribution>(discrete);

    auto binomial = nb::class_<BinomialDistribution, EmissionDistribution>(
        m, "Binomial", "Binomial distribution.");
    binomial.def(nb::init<int, double>(), nb::arg("n") = 10, nb::arg("p") = 0.5)
        .def_prop_rw("n", &BinomialDistribution::getN, &BinomialDistribution::setN)
        .def_prop_rw("p", &BinomialDistribution::getP, &BinomialDistribution::setP);
    bind_distribution_common<BinomialDistribution>(binomial);

    auto negative_binomial = nb::class_<NegativeBinomialDistribution, EmissionDistribution>(
        m, "NegativeBinomial", "Negative Binomial distribution.");
    negative_binomial.def(nb::init<double, double>(), nb::arg("r") = 5.0, nb::arg("p") = 0.5)
        .def_prop_rw("r", &NegativeBinomialDistribution::getR, &NegativeBinomialDistribution::setR)
        .def_prop_rw("p", &NegativeBinomialDistribution::getP, &NegativeBinomialDistribution::setP);
    bind_distribution_common<NegativeBinomialDistribution>(negative_binomial);

    auto poisson = nb::class_<PoissonDistribution, EmissionDistribution>(
        m, "Poisson", "Poisson distribution.");
    poisson.def(nb::init<double>(), nb::arg("lam") = 1.0)
        .def_prop_rw("lam", &PoissonDistribution::getLambda, &PoissonDistribution::setLambda);
    bind_distribution_common<PoissonDistribution>(poisson);

    auto gaussian = nb::class_<GaussianDistribution, EmissionDistribution>(
        m, "Gaussian", "Gaussian distribution.");
    gaussian.def(nb::init<double, double>(), nb::arg("mu") = 0.0, nb::arg("sigma") = 1.0)
        .def_prop_rw("mu", &GaussianDistribution::getMean, &GaussianDistribution::setMean)
        .def_prop_rw("sigma",
                     &GaussianDistribution::getStandardDeviation,
                     &GaussianDistribution::setStandardDeviation);
    bind_distribution_common<GaussianDistribution>(gaussian);

    auto exponential = nb::class_<ExponentialDistribution, EmissionDistribution>(
        m, "Exponential", "Exponential distribution.");
    exponential.def(nb::init<double>(), nb::arg("lam") = 1.0)
        .def_prop_rw("lam", &ExponentialDistribution::getLambda, &ExponentialDistribution::setLambda);
    bind_distribution_common<ExponentialDistribution>(exponential);

    auto gamma = nb::class_<GammaDistribution, EmissionDistribution>(
        m, "Gamma", "Gamma distribution.");
    gamma.def(nb::init<double, double>(), nb::arg("k") = 1.0, nb::arg("theta") = 1.0)
        .def_prop_rw("k", &GammaDistribution::getK, &GammaDistribution::setK)
        .def_prop_rw("theta", &GammaDistribution::getTheta, &GammaDistribution::setTheta);
    bind_distribution_common<GammaDistribution>(gamma);

    auto log_normal = nb::class_<LogNormalDistribution, EmissionDistribution>(
        m, "LogNormal", "Log-normal distribution.");
    log_normal.def(nb::init<double, double>(), nb::arg("mu") = 0.0, nb::arg("sigma") = 1.0)
        .def_prop_rw("mu", &LogNormalDistribution::getMean, &LogNormalDistribution::setMean)
        .def_prop_rw("sigma",
                     &LogNormalDistribution::getStandardDeviation,
                     &LogNormalDistribution::setStandardDeviation)
        .def_prop_ro("distribution_mean", &LogNormalDistribution::getDistributionMean);
    bind_distribution_common<LogNormalDistribution>(log_normal);

    auto pareto = nb::class_<ParetoDistribution, EmissionDistribution>(
        m, "Pareto", "Pareto distribution.");
    pareto.def(nb::init<double, double>(), nb::arg("k") = 1.0, nb::arg("xm") = 1.0)
        .def_prop_rw("k", &ParetoDistribution::getK, &ParetoDistribution::setK)
        .def_prop_rw("xm", &ParetoDistribution::getXm, &ParetoDistribution::setXm);
    bind_distribution_common<ParetoDistribution>(pareto);

    auto beta = nb::class_<BetaDistribution, EmissionDistribution>(
        m, "Beta", "Beta distribution.");
    beta.def(nb::init<double, double>(), nb::arg("alpha") = 1.0, nb::arg("beta") = 1.0)
        .def_prop_rw("alpha", &BetaDistribution::getAlpha, &BetaDistribution::setAlpha)
        .def_prop_rw("beta", &BetaDistribution::getBeta, &BetaDistribution::setBeta);
    bind_distribution_common<BetaDistribution>(beta);

    auto uniform = nb::class_<UniformDistribution, EmissionDistribution>(
        m, "Uniform", "Uniform distribution.");
    uniform.def(nb::init<double, double>(), nb::arg("a") = 0.0, nb::arg("b") = 1.0)
        .def_prop_rw("a", &UniformDistribution::getA, &UniformDistribution::setA)
        .def_prop_rw("b", &UniformDistribution::getB, &UniformDistribution::setB);
    bind_distribution_common<UniformDistribution>(uniform);

    auto weibull = nb::class_<WeibullDistribution, EmissionDistribution>(
        m, "Weibull", "Weibull distribution.");
    weibull.def(nb::init<double, double>(), nb::arg("k") = 1.0, nb::arg("lam") = 1.0)
        .def_prop_rw("k", &WeibullDistribution::getK, &WeibullDistribution::setK)
        .def_prop_rw("lam", &WeibullDistribution::getLambda, &WeibullDistribution::setLambda);
    bind_distribution_common<WeibullDistribution>(weibull);

    auto rayleigh = nb::class_<RayleighDistribution, EmissionDistribution>(
        m, "Rayleigh", "Rayleigh distribution.");
    rayleigh.def(nb::init<double>(), nb::arg("sigma") = 1.0)
        .def_prop_rw("sigma", &RayleighDistribution::getSigma, &RayleighDistribution::setSigma);
    bind_distribution_common<RayleighDistribution>(rayleigh);

    auto student_t = nb::class_<StudentTDistribution, EmissionDistribution>(
        m, "StudentT", "Student's t distribution.");
    student_t.def(nb::init<>())
        .def(nb::init<double>(), nb::arg("nu"))
        .def(nb::init<double, double, double>(), nb::arg("nu"), nb::arg("location"), nb::arg("scale"))
        .def_prop_rw("nu",
                     &StudentTDistribution::getDegreesOfFreedom,
                     &StudentTDistribution::setDegreesOfFreedom)
        .def_prop_rw("location", &StudentTDistribution::getLocation, &StudentTDistribution::setLocation)
        .def_prop_rw("scale", &StudentTDistribution::getScale, &StudentTDistribution::setScale);
    bind_distribution_common<StudentTDistribution>(student_t);

    auto chi_squared = nb::class_<ChiSquaredDistribution, EmissionDistribution>(
        m, "ChiSquared", "Chi-squared distribution.");
    chi_squared.def(nb::init<double>(), nb::arg("k") = 1.0)
        .def_prop_rw("k",
                     &ChiSquaredDistribution::getDegreesOfFreedom,
                     &ChiSquaredDistribution::setDegreesOfFreedom);
    bind_distribution_common<ChiSquaredDistribution>(chi_squared);

    auto von_mises = nb::class_<VonMisesDistribution, EmissionDistribution>(
        m, "VonMises",
        "Von Mises circular distribution.\n\n"
        "The canonical emission for angular observations (turning angles, wind "
        "directions).  mu is the mean direction in radians; kappa >= 0 is the "
        "concentration parameter (0 = uniform, inf = point mass at mu).");
    von_mises
        .def(nb::init<double, double>(), nb::arg("mu") = 0.0, nb::arg("kappa") = 1.0)
        .def_prop_rw("mu", &VonMisesDistribution::getMu, &VonMisesDistribution::setMu)
        .def_prop_rw("kappa", &VonMisesDistribution::getKappa, &VonMisesDistribution::setKappa)
        .def_prop_ro("circular_variance", &VonMisesDistribution::getCircularVariance);
    bind_distribution_common<VonMisesDistribution>(von_mises);
}

// ---------------------------------------------------------------------------
// bind_hmm — the HMM model class.
// ---------------------------------------------------------------------------
void bind_hmm(nb::module_ &m) {
    auto hmm = nb::class_<Hmm>(m, "Hmm", "Hidden Markov Model.");
    hmm.def(nb::init<std::size_t>(), nb::arg("num_states"))
        .def_prop_ro("num_states", &Hmm::getNumStatesModern)
        .def("validate", &Hmm::validate)
        .def("set_pi",
             [](Hmm &h, NpArray1DIn pi) {
                 if (pi.shape(0) != h.getNumStatesModern()) {
                     throw nb::value_error("pi length must match num_states");
                 }
                 h.setPi(vector_from_numpy(pi));
             },
             nb::arg("pi").noconvert())
        .def("get_pi", [](const Hmm &h) -> nb::object { return vector_to_numpy(h.getPi()); })
        .def("set_trans",
             [](Hmm &h, NpArray2DIn trans) {
                 const size_t n = h.getNumStatesModern();
                 if (trans.shape(0) != n || trans.shape(1) != n) {
                     throw nb::value_error("transition matrix must have shape (num_states, num_states)");
                 }
                 h.setTrans(matrix_from_numpy(trans));
             },
             nb::arg("trans").noconvert())
        .def("get_trans", [](const Hmm &h) -> nb::object { return matrix_to_numpy(h.getTrans()); })
        .def("set_distribution",
             [](Hmm &h, std::size_t state, const EmissionDistribution &distribution) {
                 h.setDistribution(state, clone_distribution(distribution));
             },
             nb::arg("state"),
             nb::arg("distribution"))
        // rv_policy::reference_internal: returns a raw pointer that borrows
        // from the Hmm's internal storage.  nanobind keeps the Hmm alive as
        // long as the caller holds a reference to the returned distribution.
        .def("get_distribution",
             [](Hmm &h, std::size_t state) -> EmissionDistribution * {
                 return &h.getDistribution(state);
             },
             nb::arg("state"),
             nb::rv_policy::reference_internal)
        // Returns the text representation of the distribution at 'state'.
        // For programmatic access to parameters, use get_distribution(state)
        // and read named properties directly.
        .def("get_distribution_name",
             [](const Hmm &h, std::size_t state) {
                 return h.getDistribution(state).toString();
             },
             nb::arg("state"),
             "Return the text representation of the emission distribution at state.");
}

// ---------------------------------------------------------------------------
// bind_calculators — ForwardBackwardCalculator and ViterbiCalculator.
// ---------------------------------------------------------------------------
void bind_calculators(nb::module_ &m) {
    // keep_alive<1,2>: the HMM (arg 2) is kept alive at least as long as
    // the calculator (arg 1 = self), preventing use-after-free on the
    // non-owning reference the calculator holds to its model.
    nb::class_<ForwardBackwardCalculator>(m, "ForwardBackwardCalculator")
        .def(
            "__init__",
            // Placement-new is the nanobind pattern for types whose C++
            // constructor takes non-trivial arguments.
            [](ForwardBackwardCalculator *self, const Hmm &h, NpArray1DIn observations) {
                new (self) ForwardBackwardCalculator(h, observation_set_from_numpy(observations));
            },
            nb::arg("hmm"),
            nb::arg("observations").noconvert(),
            nb::keep_alive<1, 2>())
        // GIL is released for the compute pass so other Python threads
        // can run concurrently while C++ processes the observation sequence.
        .def("compute",
             [](ForwardBackwardCalculator &calc, NpArray1DIn observations) {
                 auto obs = observation_set_from_numpy(observations);
                 nb::gil_scoped_release release;
                 calc.compute(obs);
             },
             nb::arg("observations").noconvert())
        .def("compute",
             [](ForwardBackwardCalculator &calc) {
                 nb::gil_scoped_release release;
                 calc.compute();
             })
        .def_prop_ro("log_probability", &ForwardBackwardCalculator::getLogProbability)
        .def_prop_ro("probability", &ForwardBackwardCalculator::probability)
        .def("get_log_forward_variables",
             [](const ForwardBackwardCalculator &calc) -> nb::object {
                 return matrix_to_numpy(calc.getLogForwardVariables());
             })
        .def("get_log_backward_variables",
             [](const ForwardBackwardCalculator &calc) -> nb::object {
                 return matrix_to_numpy(calc.getLogBackwardVariables());
             })
        .def_prop_ro("num_states", &ForwardBackwardCalculator::getNumStates)
        .def("decode_posterior",
             [](ForwardBackwardCalculator &calc) -> nb::object {
                 StateSequence seq;
                 {
                     nb::gil_scoped_release release;
                     seq = calc.decodePosterior();
                 }
                 return state_sequence_to_numpy(seq);
             },
             "Per-step argmax-\u03b3 decoding: returns the most probable state at each "
             "time step independently. Minimises per-step state error rate. "
             "Unlike Viterbi, the result is not guaranteed to be a valid path.");

    nb::class_<ViterbiCalculator>(m, "ViterbiCalculator")
        .def(
            "__init__",
            [](ViterbiCalculator *self, const Hmm &h, NpArray1DIn observations) {
                new (self) ViterbiCalculator(h, observation_set_from_numpy(observations));
            },
            nb::arg("hmm"),
            nb::arg("observations").noconvert(),
            nb::keep_alive<1, 2>())
        .def("decode",
             [](const ViterbiCalculator &calc) -> nb::object {
                 // The Viterbi pass ran in the constructor while the temp
                 // ObservationSet was still alive.  Re-calling calc.decode()
                 // would re-read the now-dangling obsRef_; return the cached
                 // result via getStateSequence() instead.
                 return state_sequence_to_numpy(calc.getStateSequence());
             })
        .def_prop_ro("log_probability", &ViterbiCalculator::getLogProbability)
        .def("get_state_sequence",
             [](const ViterbiCalculator &calc) -> nb::object {
                 return state_sequence_to_numpy(calc.getStateSequence());
             })
        .def_prop_ro("num_states", &ViterbiCalculator::getNumStates);
}

// ---------------------------------------------------------------------------
// bind_trainers — TrainingConfig, preset factories, and all trainer classes.
// ---------------------------------------------------------------------------
void bind_trainers(nb::module_ &m) {
    nb::class_<TrainingConfig>(m, "TrainingConfig")
        .def(nb::init<>())
        .def_rw("convergence_tolerance", &TrainingConfig::convergenceTolerance)
        .def_rw("max_iterations", &TrainingConfig::maxIterations)
        .def_rw("convergence_window", &TrainingConfig::convergenceWindow)
        .def_rw("enable_progress_reporting", &TrainingConfig::enableProgressReporting);

    m.def("training_preset_fast", []() { return training_presets::fast(); });
    m.def("training_preset_balanced", []() { return training_presets::balanced(); });
    m.def("training_preset_precise", []() { return training_presets::precise(); });

    // keep_alive<1,2> ensures the HMM outlives the holder, which in turn keeps
    // data_ alive; data_ outlives the inner trainer.
    nb::class_<BaumWelchHolder>(m, "BaumWelchTrainer")
        .def(
            "__init__",
            [](BaumWelchHolder *self, Hmm &h, const nb::list &sequences) {
                auto obs = observation_lists_from_python(sequences);
                new (self) BaumWelchHolder(h, std::move(obs));
            },
            nb::arg("hmm"),
            nb::arg("sequences"),
            nb::keep_alive<1, 2>())
        .def("train",
             [](BaumWelchHolder &h) {
                 nb::gil_scoped_release release;
                 h.trainer_.train();
             })
        .def_prop_ro("last_log_probability",
                     [](const BaumWelchHolder &h) {
                         return h.trainer_.getLastLogProbability();
                     },
                     "Total E-step log-probability from the last train() call. "
                     "-inf before train() is called or if all sequences had zero probability.");

    nb::class_<ViterbiHolder>(m, "ViterbiTrainer")
        .def(
            "__init__",
            [](ViterbiHolder *self, Hmm &h, const nb::list &sequences,
               const TrainingConfig &config) {
                auto obs = observation_lists_from_python(sequences);
                new (self) ViterbiHolder(h, std::move(obs), config);
            },
            nb::arg("hmm"),
            nb::arg("sequences"),
            nb::arg("config") = TrainingConfig{},
            nb::keep_alive<1, 2>())
        .def("train",
             [](ViterbiHolder &h) {
                 nb::gil_scoped_release release;
                 h.trainer_.train();
             })
        .def_prop_ro("has_converged",
                     [](const ViterbiHolder &h) { return h.trainer_.hasConverged(); })
        .def_prop_ro("reached_max_iterations",
                     [](const ViterbiHolder &h) { return h.trainer_.reachedMaxIterations(); })
        .def_prop_ro("last_log_probability",
                     [](const ViterbiHolder &h) { return h.trainer_.getLastLogProbability(); })
        .def_prop_rw("config",
                     [](const ViterbiHolder &h) { return h.trainer_.getConfig(); },
                     [](ViterbiHolder &h, const TrainingConfig &c) { h.trainer_.setConfig(c); });

    nb::class_<MapBaumWelchHolder>(m, "MapBaumWelchTrainer")
        .def(
            "__init__",
            [](MapBaumWelchHolder *self, Hmm &h, const nb::list &sequences,
               double pseudo_count) {
                auto obs = observation_lists_from_python(sequences);
                new (self) MapBaumWelchHolder(h, std::move(obs), pseudo_count);
            },
            nb::arg("hmm"),
            nb::arg("sequences"),
            nb::arg("pseudo_count") = 1.0,
            nb::keep_alive<1, 2>())
        .def("train",
             [](MapBaumWelchHolder &h) {
                 nb::gil_scoped_release release;
                 h.trainer_.train();
             })
        .def_prop_rw("pseudo_count",
                     [](const MapBaumWelchHolder &h) { return h.trainer_.getPseudoCount(); },
                     [](MapBaumWelchHolder &h, double c) { h.trainer_.setPseudoCount(c); })
        .def("compute_log_prior",
             [](const MapBaumWelchHolder &h) { return h.trainer_.computeLogPrior(); },
             "Unnormalised log-prior log P(\u03bb | c). "
             "Add to getLogProbability() for the correct MAP convergence criterion.");

    nb::class_<SegmentalKMeansHolder>(m, "SegmentalKMeansTrainer")
        .def(
            "__init__",
            [](SegmentalKMeansHolder *self, Hmm &h, const nb::list &sequences,
               std::size_t max_iterations) {
                auto obs = observation_lists_from_python(sequences);
                new (self) SegmentalKMeansHolder(h, std::move(obs), max_iterations);
            },
            nb::arg("hmm"),
            nb::arg("sequences"),
            nb::arg("max_iterations") = std::size_t{100},
            nb::keep_alive<1, 2>())
        .def("train",
             [](SegmentalKMeansHolder &h) {
                 nb::gil_scoped_release release;
                 h.trainer_.train();
             })
        .def_prop_ro("is_terminated",
                     [](const SegmentalKMeansHolder &h) { return h.trainer_.isTerminated(); });
}

// ---------------------------------------------------------------------------
// bind_io — JSON (recommended) and XML (legacy) HMM serialization.
// ---------------------------------------------------------------------------
void bind_io(nb::module_ &m) {
    // ---- JSON (recommended as of libhmm v3.4.0) ----
    m.def("to_json",
          [](const Hmm &hmm_model) { return libhmm::to_json(hmm_model); },
          nb::arg("hmm"),
          "Serialize an HMM to a compact JSON string.");
    m.def("from_json",
          [](const std::string &src) { return libhmm::from_json(src); },
          nb::arg("src"),
          "Deserialize an HMM from a JSON string produced by to_json(). "
          "Raises RuntimeError on malformed input.");
    m.def("save_json",
          [](const Hmm &hmm_model, const std::string &filepath) {
              libhmm::save_json(hmm_model, std::filesystem::path{filepath});
          },
          nb::arg("hmm"),
          nb::arg("filepath"),
          "Write an HMM as JSON to filepath.");
    m.def("load_json",
          [](const std::string &filepath) {
              return libhmm::load_json(std::filesystem::path{filepath});
          },
          nb::arg("filepath"),
          "Read and deserialize an HMM from a JSON file.");

    // ---- XML (legacy; deprecated in libhmm v3.4.0, retained for existing files) ----
    m.def("load_hmm",
          [](const std::string &filepath) {
              XMLFileReader reader;
              return reader.read(std::filesystem::path{filepath});
          },
          nb::arg("filepath"),
          "Load an HMM from a legacy XML file. Prefer load_json() for new code.");
    m.def("save_hmm",
          [](const Hmm &hmm_model, const std::string &filepath) {
              XMLFileWriter writer;
              writer.write(hmm_model, std::filesystem::path{filepath});
          },
          nb::arg("hmm"),
          nb::arg("filepath"),
          "Save an HMM to a legacy XML file. Prefer save_json() for new code.");
}

// ---------------------------------------------------------------------------
// bind_model_selection — AIC, BIC, AICc, and parameter counting.
// ---------------------------------------------------------------------------
void bind_model_selection(nb::module_ &m) {
    m.def("count_free_parameters",
          [](const Hmm &hmm) { return libhmm::count_free_parameters(hmm); },
          nb::arg("hmm"),
          "Count the free parameters of a fitted HMM: N(N\u22121) transitions + (N\u22121) initial "
          "+ \u03a3 emission params.");
    m.def("compute_aic",
          [](double logL, std::size_t k) { return libhmm::compute_aic(logL, k); },
          nb::arg("log_likelihood"),
          nb::arg("k"),
          "AIC = 2k \u2212 2 logL (lower is better).");
    m.def("compute_bic",
          [](double logL, std::size_t k, std::size_t n) {
              return libhmm::compute_bic(logL, k, n);
          },
          nb::arg("log_likelihood"),
          nb::arg("k"),
          nb::arg("n"),
          "BIC = k ln(n) \u2212 2 logL (lower is better).");
    m.def("compute_aicc",
          [](double logL, std::size_t k, std::size_t n) {
              return libhmm::compute_aicc(logL, k, n);
          },
          nb::arg("log_likelihood"),
          nb::arg("k"),
          nb::arg("n"),
          "AICc = AIC + 2k(k+1)/(n\u2212k\u22121) (corrected AIC; lower is better).");
    m.def("evaluate_model",
          [](const Hmm &hmm, double logL, std::size_t n) -> nb::tuple {
              const auto c = libhmm::evaluate_model(hmm, logL, n);
              return nb::make_tuple(c.aic, c.bic, c.aicc);
          },
          nb::arg("hmm"),
          nb::arg("log_likelihood"),
          nb::arg("sequence_length"),
          "Convenience wrapper: returns (aic, bic, aicc) for the fitted HMM.");
}

// ---------------------------------------------------------------------------
// MV clone registry and helper.
// ---------------------------------------------------------------------------
using MVEmissionDist = BasicEmissionDistribution<ObservationVectorView>;
using MVCloneFn = std::function<std::unique_ptr<MVEmissionDist>(const MVEmissionDist &)>;

template <typename Dist>
MVCloneFn make_mv_clone() {
    return [](const MVEmissionDist &base) -> std::unique_ptr<MVEmissionDist> {
        return std::make_unique<Dist>(static_cast<const Dist &>(base));
    };
}

const std::unordered_map<std::type_index, MVCloneFn> &mv_clone_registry() {
    static const std::unordered_map<std::type_index, MVCloneFn> reg{
        {typeid(DiagonalGaussianDistribution),       make_mv_clone<DiagonalGaussianDistribution>()},
        {typeid(FullCovarianceGaussianDistribution), make_mv_clone<FullCovarianceGaussianDistribution>()},
        {typeid(IndependentComponentsDistribution),  make_mv_clone<IndependentComponentsDistribution>()},
    };
    return reg;
}

[[nodiscard]] std::unique_ptr<MVEmissionDist>
clone_mv_distribution(const MVEmissionDist &dist) {
    const auto it = mv_clone_registry().find(typeid(dist));
    if (it == mv_clone_registry().end())
        throw nb::type_error("Unsupported MV EmissionDistribution subtype");
    return it->second(dist);
}

// ---------------------------------------------------------------------------
// bind_mv_distributions
// ---------------------------------------------------------------------------
void bind_mv_distributions(nb::module_ &m) {
    // Shared MV base (not typically instantiated directly from Python)
    nb::class_<MVEmissionDist>(m, "MVEmissionDistribution");

    // DiagonalGaussianDistribution
    auto diag = nb::class_<DiagonalGaussianDistribution, MVEmissionDist>(
        m, "DiagonalGaussian",
        "Multivariate Gaussian with diagonal covariance (D independent Gaussians).\n\n"
        "Observation vector must be a 1-D float64 array of length dim for log_pdf,\n"
        "or a 2-D (N×dim) array for batched log_pdf.");
    diag.def(nb::init<std::size_t>(), nb::arg("dim"))
        .def_prop_ro("dim", &DiagonalGaussianDistribution::getDimension)
        .def_prop_ro("means",
                     [](const DiagonalGaussianDistribution &d) -> nb::object {
                         const auto &v = d.getMean();
                         auto ptr = std::make_unique<double[]>(v.size());
                         std::copy(v.begin(), v.end(), ptr.get());
                         auto *raw = ptr.release();
                         return buf_to_numpy_owned(raw, v.size());
                     })
        .def_prop_ro("variances",
                     [](const DiagonalGaussianDistribution &d) -> nb::object {
                         const auto &v = d.getVariance();
                         auto ptr = std::make_unique<double[]>(v.size());
                         std::copy(v.begin(), v.end(), ptr.get());
                         auto *raw = ptr.release();
                         return buf_to_numpy_owned(raw, v.size());
                     })
        .def("set_parameters",
             [](DiagonalGaussianDistribution &d, NpArray1DIn means, NpArray1DIn variances) {
                 std::vector<double> mu(means.data(), means.data() + means.shape(0));
                 std::vector<double> var(variances.data(), variances.data() + variances.shape(0));
                 d.setParameters(std::move(mu), var);
             },
             nb::arg("means").noconvert(), nb::arg("variances").noconvert())
        .def("set_means",
             [](DiagonalGaussianDistribution &d, NpArray1DIn means) {
                 std::vector<double> mu(means.data(), means.data() + means.shape(0));
                 d.setMeans(std::move(mu));
             }, nb::arg("means").noconvert())
        .def("set_variances",
             [](DiagonalGaussianDistribution &d, NpArray1DIn vars) {
                 std::vector<double> v(vars.data(), vars.data() + vars.shape(0));
                 d.setVariances(v);
             }, nb::arg("variances").noconvert())
        .def("log_pdf",
             [](const DiagonalGaussianDistribution &d, NpArray1DIn x) -> double {
                 return d.getLogProbability(ObservationVectorView{x.data(), x.shape(0)});
             }, nb::arg("x").noconvert(), "Log-pdf for a single D-dim observation.")
        .def("log_pdf",
             [](const DiagonalGaussianDistribution &d, NpArray2DIn X) -> nb::object {
                 const size_t N = X.shape(0), D = X.shape(1);
                 auto *buf = new double[N];
                 { nb::gil_scoped_release rel;
                   for (size_t i = 0; i < N; ++i)
                       buf[i] = d.getLogProbability(
                           ObservationVectorView{X.data() + i * D, D}); }
                 return buf_to_numpy_owned(buf, N);
             }, nb::arg("X").noconvert(), "Batched log-pdf for a 2-D (N×D) observation array.")
        .def("fit",
             [](DiagonalGaussianDistribution &d, NpArray2DIn X) {
                 auto views = obs_matrix_views(X);
                 nb::gil_scoped_release rel;
                 d.fit(std::span<const ObservationVectorView>(views.data(), views.size()));
             }, nb::arg("X").noconvert())
        .def("fit_weighted",
             [](DiagonalGaussianDistribution &d, NpArray2DIn X, NpArray1DIn w) {
                 auto views = obs_matrix_views(X);
                 nb::gil_scoped_release rel;
                 d.fit(std::span<const ObservationVectorView>(views.data(), views.size()),
                       std::span<const double>(w.data(), w.shape(0)));
             }, nb::arg("X").noconvert(), nb::arg("weights").noconvert())
        .def("reset", &DiagonalGaussianDistribution::reset)
        .def("sample_mv",
             [](const DiagonalGaussianDistribution &d) -> nb::object {
                 auto v = d.sample_mv(g_rng);
                 auto ptr = std::make_unique<double[]>(v.size());
                 std::copy(v.begin(), v.end(), ptr.get());
                 auto *raw = ptr.release();
                 return buf_to_numpy_owned(raw, v.size());
             })
        .def("__repr__", [](const DiagonalGaussianDistribution &d){ return d.toString(); });

    // FullCovarianceGaussianDistribution
    auto full = nb::class_<FullCovarianceGaussianDistribution, MVEmissionDist>(
        m, "FullCovGaussian",
        "Multivariate Gaussian with full covariance matrix.\n\n"
        "Observation vector must be a 1-D float64 array of length dim for log_pdf,\n"
        "or a 2-D (N×dim) array for batched log_pdf.");
    full.def(nb::init<std::size_t>(), nb::arg("dim"))
        .def_prop_ro("dim", &FullCovarianceGaussianDistribution::getDimension)
        .def_prop_ro("mean",
                     [](const FullCovarianceGaussianDistribution &d) -> nb::object {
                         const auto &v = d.getMean();
                         auto ptr = std::make_unique<double[]>(v.size());
                         std::copy(v.begin(), v.end(), ptr.get());
                         auto *raw = ptr.release();
                         return buf_to_numpy_owned(raw, v.size());
                     })
        .def_prop_ro("covariance",
                     [](const FullCovarianceGaussianDistribution &d) -> nb::object {
                         return obs_matrix_to_numpy(d.getCovariance());
                     })
        .def("set_mean",
             [](FullCovarianceGaussianDistribution &d, NpArray1DIn mu) {
                 std::vector<double> v(mu.data(), mu.data() + mu.shape(0));
                 d.setMean(std::move(v));
             }, nb::arg("mean").noconvert())
        .def("set_covariance",
             [](FullCovarianceGaussianDistribution &d, NpArray2DIn cov) {
                 d.setCovariance(obs_matrix_from_numpy(cov));
             }, nb::arg("cov").noconvert())
        .def("set_parameters",
             [](FullCovarianceGaussianDistribution &d, NpArray1DIn mu, NpArray2DIn cov) {
                 std::vector<double> v(mu.data(), mu.data() + mu.shape(0));
                 d.setParameters(std::move(v), obs_matrix_from_numpy(cov));
             }, nb::arg("mean").noconvert(), nb::arg("cov").noconvert())
        .def("log_pdf",
             [](const FullCovarianceGaussianDistribution &d, NpArray1DIn x) -> double {
                 return d.getLogProbability(ObservationVectorView{x.data(), x.shape(0)});
             }, nb::arg("x").noconvert(), "Log-pdf for a single D-dim observation.")
        .def("log_pdf",
             [](const FullCovarianceGaussianDistribution &d, NpArray2DIn X) -> nb::object {
                 const size_t N = X.shape(0), D = X.shape(1);
                 auto *buf = new double[N];
                 { nb::gil_scoped_release rel;
                   for (size_t i = 0; i < N; ++i)
                       buf[i] = d.getLogProbability(
                           ObservationVectorView{X.data() + i * D, D}); }
                 return buf_to_numpy_owned(buf, N);
             }, nb::arg("X").noconvert(), "Batched log-pdf for a 2-D (N×D) observation array.")
        .def("fit",
             [](FullCovarianceGaussianDistribution &d, NpArray2DIn X) {
                 auto views = obs_matrix_views(X);
                 nb::gil_scoped_release rel;
                 d.fit(std::span<const ObservationVectorView>(views.data(), views.size()));
             }, nb::arg("X").noconvert())
        .def("fit_weighted",
             [](FullCovarianceGaussianDistribution &d, NpArray2DIn X, NpArray1DIn w) {
                 auto views = obs_matrix_views(X);
                 nb::gil_scoped_release rel;
                 d.fit(std::span<const ObservationVectorView>(views.data(), views.size()),
                       std::span<const double>(w.data(), w.shape(0)));
             }, nb::arg("X").noconvert(), nb::arg("weights").noconvert())
        .def("reset", &FullCovarianceGaussianDistribution::reset)
        .def("sample_mv",
             [](const FullCovarianceGaussianDistribution &d) -> nb::object {
                 auto v = d.sample_mv(g_rng);
                 auto ptr = std::make_unique<double[]>(v.size());
                 std::copy(v.begin(), v.end(), ptr.get());
                 auto *raw = ptr.release();
                 return buf_to_numpy_owned(raw, v.size());
             })
        .def("__repr__", [](const FullCovarianceGaussianDistribution &d){ return d.toString(); });

    // IndependentComponentsDistribution
    auto ic = nb::class_<IndependentComponentsDistribution, MVEmissionDist>(
        m, "IndependentComponents",
        "MV emission with D independent scalar components.\n\n"
        "Default construction creates D Gaussian(0,1) components.\n"
        "Individual components can be any scalar EmissionDistribution.");
    ic.def(nb::init<std::size_t>(), nb::arg("dim"),
           "Construct with dim independent Gaussian(0,1) components.")
      .def("__init__",
           [](IndependentComponentsDistribution *self, const nb::list &components) {
               std::vector<std::unique_ptr<EmissionDistribution>> comps;
               comps.reserve(nb::len(components));
               for (nb::handle item : components)
                   comps.push_back(clone_distribution(
                       nb::cast<const EmissionDistribution &>(item)));
               new (self) IndependentComponentsDistribution(std::move(comps));
           },
           nb::arg("components"),
           "Construct from a list of scalar EmissionDistribution objects.")
      .def_prop_ro("dim", &IndependentComponentsDistribution::getDimension)
      .def("get_component",
           [](IndependentComponentsDistribution &d, std::size_t i) -> EmissionDistribution & {
               return d.getComponent(i);
           }, nb::arg("index"), nb::rv_policy::reference_internal)
      .def("set_component",
           [](IndependentComponentsDistribution &d, std::size_t i,
              const EmissionDistribution &comp) {
               d.setComponent(i, clone_distribution(comp));
           }, nb::arg("index"), nb::arg("distribution"))
      .def("log_pdf",
           [](const IndependentComponentsDistribution &d, NpArray1DIn x) -> double {
               return d.getLogProbability(ObservationVectorView{x.data(), x.shape(0)});
           }, nb::arg("x").noconvert())
      .def("log_pdf",
           [](const IndependentComponentsDistribution &d, NpArray2DIn X) -> nb::object {
               const size_t N = X.shape(0), D = X.shape(1);
               auto *buf = new double[N];
               { nb::gil_scoped_release rel;
                 for (size_t i = 0; i < N; ++i)
                     buf[i] = d.getLogProbability(
                         ObservationVectorView{X.data() + i * D, D}); }
               return buf_to_numpy_owned(buf, N);
           }, nb::arg("X").noconvert())
      .def("fit",
           [](IndependentComponentsDistribution &d, NpArray2DIn X) {
               auto views = obs_matrix_views(X);
               nb::gil_scoped_release rel;
               d.fit(std::span<const ObservationVectorView>(views.data(), views.size()));
           }, nb::arg("X").noconvert())
      .def("fit_weighted",
           [](IndependentComponentsDistribution &d, NpArray2DIn X, NpArray1DIn w) {
               auto views = obs_matrix_views(X);
               nb::gil_scoped_release rel;
               d.fit(std::span<const ObservationVectorView>(views.data(), views.size()),
                     std::span<const double>(w.data(), w.shape(0)));
           }, nb::arg("X").noconvert(), nb::arg("weights").noconvert())
      .def("reset", &IndependentComponentsDistribution::reset)
      .def("__repr__",
           [](const IndependentComponentsDistribution &d){ return d.toString(); });
}

// ---------------------------------------------------------------------------
// bind_hmm_mv — the multivariate HMM.
// ---------------------------------------------------------------------------
void bind_hmm_mv(nb::module_ &m) {
    auto hmm_mv = nb::class_<HmmMV>(m, "HmmMV",
        "Multivariate Hidden Markov Model (BasicHmm<ObservationVectorView>).\n\n"
        "Emission distributions must be MV types (DiagonalGaussian, FullCovGaussian, "
        "or IndependentComponents).\n"
        "Observation sequences are 2-D float64 NumPy arrays with shape (T, D).");
    hmm_mv
        .def(nb::init<std::size_t>(), nb::arg("num_states"))
        .def_prop_ro("num_states", &HmmMV::getNumStatesModern)
        .def("set_pi",
             [](HmmMV &h, NpArray1DIn pi) {
                 if (pi.shape(0) != h.getNumStatesModern())
                     throw nb::value_error("pi length must match num_states");
                 h.setPi(vector_from_numpy(pi));
             }, nb::arg("pi").noconvert())
        .def("get_pi",
             [](const HmmMV &h) -> nb::object { return vector_to_numpy(h.getPi()); })
        .def("set_trans",
             [](HmmMV &h, NpArray2DIn trans) {
                 const size_t n = h.getNumStatesModern();
                 if (trans.shape(0) != n || trans.shape(1) != n)
                     throw nb::value_error("transition matrix must have shape (num_states, num_states)");
                 h.setTrans(matrix_from_numpy(trans));
             }, nb::arg("trans").noconvert())
        .def("get_trans",
             [](const HmmMV &h) -> nb::object { return matrix_to_numpy(h.getTrans()); })
        .def("set_distribution",
             [](HmmMV &h, std::size_t state, const MVEmissionDist &dist) {
                 h.setDistribution(state, clone_mv_distribution(dist));
             }, nb::arg("state"), nb::arg("distribution"))
        .def("get_distribution",
             [](HmmMV &h, std::size_t state) -> MVEmissionDist & {
                 return h.getDistribution(state);
             }, nb::arg("state"), nb::rv_policy::reference_internal)
        .def("validate", &HmmMV::validate);
}

// ---------------------------------------------------------------------------
// bind_mv_calculators
// ---------------------------------------------------------------------------
void bind_mv_calculators(nb::module_ &m) {
    using MVFBC = BasicForwardBackwardCalculator<ObservationVectorView>;
    nb::class_<MVFBC>(m, "MVForwardBackwardCalculator")
        .def("__init__",
             [](MVFBC *self, HmmMV &h, NpArray2DIn observations) {
                 auto mat = obs_matrix_from_numpy(observations);
                 new (self) MVFBC(h, std::move(mat));
             },
             nb::arg("hmm"), nb::arg("observations").noconvert(),
             nb::keep_alive<1, 2>())
        .def_prop_ro("log_probability", &MVFBC::getLogProbability)
        .def("get_log_forward_variables",
             [](const MVFBC &calc) -> nb::object {
                 return matrix_to_numpy(calc.getLogForwardVariables());
             })
        .def("get_log_backward_variables",
             [](const MVFBC &calc) -> nb::object {
                 return matrix_to_numpy(calc.getLogBackwardVariables());
             })
        .def_prop_ro("num_states", &MVFBC::getNumStates)
        .def("decode_posterior",
             [](MVFBC &calc) -> nb::object {
                 StateSequence seq;
                 {
                     nb::gil_scoped_release release;
                     seq = calc.decodePosterior();
                 }
                 return state_sequence_to_numpy(seq);
             },
             "Per-step argmax-\u03b3 decoding: returns the most probable state at each "
             "time step independently. Minimises per-step state error rate. "
             "Unlike Viterbi, the result is not guaranteed to be a valid path.");

    using MVVC = BasicViterbiCalculator<ObservationVectorView>;
    nb::class_<MVVC>(m, "MVViterbiCalculator")
        .def("__init__",
             [](MVVC *self, HmmMV &h, NpArray2DIn observations) {
                 auto mat = obs_matrix_from_numpy(observations);
                 new (self) MVVC(h, std::move(mat));
             },
             nb::arg("hmm"), nb::arg("observations").noconvert(),
             nb::keep_alive<1, 2>())
        .def_prop_ro("log_probability", &MVVC::getLogProbability)
        .def("decode",
             [](MVVC &calc) -> nb::object {
                 StateSequence seq;
                 {
                     nb::gil_scoped_release release;
                     seq = calc.decode();
                 }
                 return state_sequence_to_numpy(seq);
             },
             "Viterbi MAP decoding: returns the most probable state sequence (1-D int64 array). "
             "Use when whole-sequence structural coherence is required.")
        .def_prop_ro("num_states", &MVVC::getNumStates);
}

// ---------------------------------------------------------------------------
// bind_mv_trainers
// ---------------------------------------------------------------------------
void bind_mv_trainers(nb::module_ &m) {
    nb::class_<MVBaumWelchHolder>(m, "MVBaumWelchTrainer")
        .def("__init__",
             [](MVBaumWelchHolder *self, HmmMV &h, const nb::list &sequences) {
                 auto lists = multi_obs_lists_from_python(sequences);
                 new (self) MVBaumWelchHolder(h, std::move(lists));
             },
             nb::arg("hmm"), nb::arg("sequences"),
             nb::keep_alive<1, 2>())
        .def("train",
             [](MVBaumWelchHolder &h) {
                 nb::gil_scoped_release release;
                 h.trainer_.train();
             })
        .def_prop_ro("last_log_probability",
                     [](const MVBaumWelchHolder &h) {
                         return h.trainer_.getLastLogProbability();
                     },
                     "Total E-step log-probability from the last train() call. "
                     "-inf before train() is called or if all sequences had zero probability.");

    nb::class_<MVSegmentalKMeansHolder>(m, "MVSegmentalKMeansTrainer")
        .def("__init__",
             [](MVSegmentalKMeansHolder *self, HmmMV &h, const nb::list &sequences,
                std::size_t max_iterations) {
                 auto lists = multi_obs_lists_from_python(sequences);
                 new (self) MVSegmentalKMeansHolder(h, std::move(lists), max_iterations);
             },
             nb::arg("hmm"),
             nb::arg("sequences"),
             nb::arg("max_iterations") = std::size_t{100},
             nb::keep_alive<1, 2>())
        .def("train",
             [](MVSegmentalKMeansHolder &h) {
                 nb::gil_scoped_release release;
                 h.trainer_.train();
             })
        .def_prop_ro("is_terminated",
                     [](const MVSegmentalKMeansHolder &h) {
                         return h.trainer_.isTerminated();
                     },
                     "True if the last train() call converged (no assignment change). "
                     "False if max_iterations was reached without convergence.");

    nb::class_<MVMapBaumWelchHolder>(m, "MVMapBaumWelchTrainer")
        .def("__init__",
             [](MVMapBaumWelchHolder *self, HmmMV &h, const nb::list &sequences,
                double pseudo_count) {
                 auto lists = multi_obs_lists_from_python(sequences);
                 new (self) MVMapBaumWelchHolder(h, std::move(lists), pseudo_count);
             },
             nb::arg("hmm"), nb::arg("sequences"),
             nb::arg("pseudo_count") = 1.0,
             nb::keep_alive<1, 2>())
        .def("train",
             [](MVMapBaumWelchHolder &h) {
                 nb::gil_scoped_release release;
                 h.trainer_.train();
             })
        .def_prop_rw("pseudo_count",
                     [](const MVMapBaumWelchHolder &h) { return h.trainer_.getPseudoCount(); },
                     [](MVMapBaumWelchHolder &h, double c) { h.trainer_.setPseudoCount(c); })
        .def("compute_log_prior",
             [](const MVMapBaumWelchHolder &h) { return h.trainer_.computeLogPrior(); },
             "Unnormalised log-prior log P(\u03bb | c). "
             "Add to last_log_probability for the correct MAP convergence criterion.");

    m.def("kmeans_init",
          [](HmmMV &hmm, const nb::list &sequences, uint64_t seed) {
              auto lists  = multi_obs_lists_from_python(sequences);
              std::mt19937_64 rng{seed};
              nb::gil_scoped_release release;
              libhmm::kmeans_init(hmm, lists, rng);
          },
          nb::arg("hmm"),
          nb::arg("sequences"),
          nb::arg("seed") = uint64_t{42},
          "Initialise HmmMV emission distributions via k-means++ on all observations.\n"
          "sequences: list of 2-D (T×D) float64 NumPy arrays.");
}

// ---------------------------------------------------------------------------
// bind_mv_io — MV JSON serialization and model selection.
// ---------------------------------------------------------------------------
void bind_mv_io(nb::module_ &m) {
    m.def("to_json_mv",
          [](const HmmMV &hmm) { return libhmm::to_json(hmm); },
          nb::arg("hmm"),
          "Serialize an MV HMM to a compact JSON string (obs_type=\"multivariate\").");
    m.def("from_json_mv",
          [](const std::string &src) { return libhmm::from_json_mv(src); },
          nb::arg("src"),
          "Deserialize an MV HMM from a JSON string produced by to_json_mv().");
    m.def("save_json_mv",
          [](const HmmMV &hmm, const std::string &filepath) {
              libhmm::save_json_mv(hmm, std::filesystem::path{filepath});
          },
          nb::arg("hmm"), nb::arg("filepath"),
          "Write an MV HMM as JSON to filepath.");
    m.def("load_json_mv",
          [](const std::string &filepath) {
              return libhmm::load_json_mv(std::filesystem::path{filepath});
          },
          nb::arg("filepath"),
          "Read and deserialize an MV HMM from a JSON file.");
    m.def("count_free_parameters_mv",
          [](const HmmMV &hmm) { return libhmm::count_free_parameters(hmm); },
          nb::arg("hmm"),
          "Count the free parameters of a fitted MV HMM.");
}

} // namespace

NB_MODULE(_core, m) {
    m.doc() = "pylibhmm native extension module";
    bind_distributions(m);
    bind_hmm(m);
    bind_calculators(m);
    bind_trainers(m);
    bind_io(m);
    bind_model_selection(m);
    // v4 multivariate API
    bind_mv_distributions(m);
    bind_hmm_mv(m);
    bind_mv_calculators(m);
    bind_mv_trainers(m);
    bind_mv_io(m);
}
