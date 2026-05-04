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
#include <functional>
#include <memory>
#include <span>
#include <type_traits>
#include <typeindex>
#include <unordered_map>

#include <libhmm/calculators/forward_backward_calculator.h>
#include <libhmm/calculators/viterbi_calculator.h>
#include <libhmm/distributions/beta_distribution.h>
#include <libhmm/distributions/binomial_distribution.h>
#include <libhmm/distributions/chi_squared_distribution.h>
#include <libhmm/distributions/discrete_distribution.h>
#include <libhmm/distributions/emission_distribution.h>
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
#include <libhmm/distributions/weibull_distribution.h>
#include <libhmm/hmm.h>
#include <libhmm/io/xml_file_reader.h>
#include <libhmm/io/xml_file_writer.h>
#include <libhmm/training/baum_welch_trainer.h>
#include <libhmm/training/segmental_kmeans_trainer.h>
#include <libhmm/training/viterbi_trainer.h>

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
        .def("reset", &Dist::reset, "Reset distribution parameters to defaults.")
        .def_prop_ro("is_discrete", &Dist::isDiscrete)
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
// bind_distributions — base EmissionDistribution and all concrete subtypes.
// ---------------------------------------------------------------------------
void bind_distributions(nb::module_ &m) {
    auto emission = nb::class_<EmissionDistribution>(m, "EmissionDistribution");
    emission.def("pdf", &EmissionDistribution::getProbability, nb::arg("x"))
        .def("log_pdf", &EmissionDistribution::getLogProbability, nb::arg("x"))
        .def("reset", &EmissionDistribution::reset)
        .def_prop_ro("is_discrete", &EmissionDistribution::isDiscrete)
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
        .def("get_distribution_name",
             [](const Hmm &h, std::size_t state) {
                 return h.getDistribution(state).toString();
             },
             nb::arg("state"));
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
        .def_prop_ro("num_states", &ForwardBackwardCalculator::getNumStates);

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
             [](ViterbiCalculator &calc) -> nb::object {
                 StateSequence seq;
                 {
                     nb::gil_scoped_release release;
                     seq = calc.decode();
                 }
                 return state_sequence_to_numpy(seq);
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

    // keep_alive<1,2> ensures the HMM outlives the trainer, which holds a
    // non-owning reference through potentially many training iterations.
    nb::class_<BaumWelchTrainer>(m, "BaumWelchTrainer")
        .def(
            "__init__",
            [](BaumWelchTrainer *self, Hmm &h, const nb::list &sequences) {
                auto obs = observation_lists_from_python(sequences);
                new (self) BaumWelchTrainer(h, obs);
            },
            nb::arg("hmm"),
            nb::arg("sequences"),
            nb::keep_alive<1, 2>())
        .def("train",
             // Release the GIL: a full EM iteration can be expensive and
             // should not block other Python threads for its duration.
             [](BaumWelchTrainer &trainer) {
                 nb::gil_scoped_release release;
                 trainer.train();
             });

    nb::class_<ViterbiTrainer>(m, "ViterbiTrainer")
        .def(
            "__init__",
            [](ViterbiTrainer *self, Hmm &h, const nb::list &sequences, const TrainingConfig &config) {
                auto obs = observation_lists_from_python(sequences);
                new (self) ViterbiTrainer(h, obs, config);
            },
            nb::arg("hmm"),
            nb::arg("sequences"),
            nb::arg("config") = TrainingConfig{},
            nb::keep_alive<1, 2>())
        .def("train",
             [](ViterbiTrainer &trainer) {
                 nb::gil_scoped_release release;
                 trainer.train();
             })
        .def_prop_ro("has_converged", &ViterbiTrainer::hasConverged)
        .def_prop_ro("reached_max_iterations", &ViterbiTrainer::reachedMaxIterations)
        .def_prop_ro("last_log_probability", &ViterbiTrainer::getLastLogProbability)
        .def_prop_rw("config", &ViterbiTrainer::getConfig, &ViterbiTrainer::setConfig);

    nb::class_<SegmentalKMeansTrainer>(m, "SegmentalKMeansTrainer")
        .def(
            "__init__",
            [](SegmentalKMeansTrainer *self, Hmm &h, const nb::list &sequences) {
                auto obs = observation_lists_from_python(sequences);
                new (self) SegmentalKMeansTrainer(h, obs);
            },
            nb::arg("hmm"),
            nb::arg("sequences"),
            nb::keep_alive<1, 2>())
        .def("train",
             [](SegmentalKMeansTrainer &trainer) {
                 nb::gil_scoped_release release;
                 trainer.train();
             })
        .def_prop_ro("is_terminated", &SegmentalKMeansTrainer::isTerminated);
}

// ---------------------------------------------------------------------------
// bind_io — XML file-based HMM serialization.
// ---------------------------------------------------------------------------
void bind_io(nb::module_ &m) {
    m.def("load_hmm",
          [](const std::string &filepath) {
              XMLFileReader reader;
              return reader.read(filepath);
          },
          nb::arg("filepath"),
          "Load an HMM from an XML file.");
    m.def("save_hmm",
          [](const Hmm &hmm_model, const std::string &filepath) {
              XMLFileWriter writer;
              writer.write(hmm_model, filepath);
          },
          nb::arg("hmm"),
          nb::arg("filepath"),
          "Save an HMM to an XML file.");
}

} // namespace

NB_MODULE(_core, m) {
    m.doc() = "pylibhmm native extension module";
    bind_distributions(m);
    bind_hmm(m);
    bind_calculators(m);
    bind_trainers(m);
    bind_io(m);
}
