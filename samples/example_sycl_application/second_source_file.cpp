#include <CL/sycl.hpp>

#include <iostream>

using namespace cl::sycl;

extern const size_t M;
extern const size_t N;

void temp(queue & myQueue,buffer<float, 2> & a,buffer<float, 2> & b, buffer<float, 2> & c)
{

    myQueue.submit([&](handler& cgh) {
      auto A = a.get_access<access::mode::read>(cgh);
      auto B = b.get_access<access::mode::read>(cgh);
      auto C = c.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class matrix_add>(
          range<2>{N, M}, [=](id<2> index) { C[index] = A[index] + B[index]; });
    });

}
