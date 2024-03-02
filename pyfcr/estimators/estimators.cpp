#include "src/biometrics.hpp"
#include "src/assaf.hpp"


PYBIND11_MODULE(_estimators, m) {
    m.doc() = "A frailty computational library";
    m.def("calculate_frailty_exponent_estimators_assaf_C", &Assaf::calculate_frailty_exponent_estimators_assaf_C);
    m.def("calculate_frailty_covariance_estimators_assaf_c", &Assaf::calculate_frailty_covariance_estimators_assaf_C);
    m.def("calculate_frailty_exponent_estimators_biometrics_c", &Biometrics::calculate_frailty_exponent_estimators_biometrics_C);
    m.def("calculate_frailty_covariance_estimators_biometrics_c", &Biometrics::calculate_frailty_covariance_estimators_biometrics_C);

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
