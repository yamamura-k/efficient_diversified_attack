from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

source = ["C/_cluster_coef.c", "./cluster_coef.pyx"]

setup(
    cmdclass=dict(build_ext=build_ext),
    ext_modules=[Extension("cluster_coef", source, language="c")],
    include_dirs=[np.get_include()],
)

source = ["CPP/_cluster_coef_2.cpp", "./cluster_coef_2.pyx"]

setup(
    cmdclass=dict(build_ext=build_ext),
    ext_modules=[Extension("cluster_coef_2", source, language="c++", extra_compile_args=["-std=c++11"])],
    include_dirs=[np.get_include()],
)

source = ["CPP/_cluster_coef_para.cpp", "./cluster_coef_para.pyx"]

setup(
    cmdclass=dict(build_ext=build_ext),
    ext_modules=[
        Extension(
            "cluster_coef_para",
            source,
            language="c++",
            extra_compile_args=["-std=c++11", "-fopenmp"],
            extra_link_args=["-lgomp"],
        )
    ],
    include_dirs=[np.get_include()],
)
source = ["CPP/_cluster_coef_para_nodeweight.cpp", "./cluster_coef_para_nodeweights.pyx"]

setup(
    cmdclass=dict(build_ext=build_ext),
    ext_modules=[
        Extension(
            "cluster_coef_para_nodeweight",
            source,
            language="c++",
            extra_compile_args=["-std=c++11", "-fopenmp"],
            extra_link_args=["-lgomp"],
        )
    ],
    include_dirs=[np.get_include()],
)

source = ["count_target_classes.pyx"]
setup(
    cmdclass=dict(build_ext=build_ext),
    ext_modules=[Extension("count_target_classes", source)],
    include_dirs=[np.get_include()],
)
