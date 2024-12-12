#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <Kokkos_Core.hpp>

void Init() {
#ifdef USE_GPU
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  auto settings = Kokkos::InitializationSettings()
                      .set_num_threads(10)
                      .set_device_id(0)
                      .set_disable_warnings(false);
#else
  auto settings =
      Kokkos::InitializationSettings().set_num_threads(10).set_disable_warnings(
          false);
#endif

  Kokkos::initialize(settings);
}

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

PYBIND11_MODULE(hignn, m) {
  m.doc() = R"pbdoc(
        Hydrodynamic Interaction Graph Neural Network
    )pbdoc";

  m.def("Init", &Init, "Initialize Kokkos");
  m.def("Finalize", &Kokkos::finalize, "Finalize Kokkos");
}