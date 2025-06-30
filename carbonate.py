#"""
#   :synopsis: Driver run file for TPL example
#   :version: 2.0
#   :maintainer: Jeffrey Hyman
#.. moduleauthor:: Jeffrey Hyman <jhyman@lanl.gov>
#"""

from pydfnworks import *
import os

jobname = os.getcwd() + "/output"

DFN = DFNWORKS(jobname,
               ncpu=8)

DFN.params['domainSize']['value'] = [25, 10, 10]
DFN.params['h']['value'] = 0.01
DFN.params['domainSizeIncrease']['value'] = [2.5, 2.5, 2.5]
DFN.params['keepOnlyLargestCluster']['value'] = True
DFN.params['ignoreBoundaryFaces']['value'] = False
DFN.params['boundaryFaces']['value'] = [1, 1, 0, 0, 0, 0]
DFN.params['seed']['value'] = 1
DFN.params['tripleIntersections']['value'] = True 
DFN.params['disableFram']['value'] = True
DFN.params['orientationOption']['value'] = 1

DFN.add_fracture_family(shape="rect",
                        distribution="tpl",
                        alpha=2.162,
                        min_radius=1,
                        max_radius=5,
                        kappa=39.9,
                        plunge=86.0,
                        trend=330.0,
                        aspect=2,
                        p32=2.52,
                        hy_variable='aperture',
                        hy_function='constant',
                        hy_params={
                            "mu": 2.2274e-6,
                        })

DFN.add_fracture_family(shape="rect",
                        distribution="tpl",
                        alpha=1.31,
                        min_radius=1,
                        max_radius=5,
                        kappa=61.3,
                        plunge=87.0,
                        trend=337.0,
                        aspect=2,
                        p32=4.84,
                        hy_variable='aperture',
                        hy_function='constant',
                        hy_params={
                            "mu": 9.019e-6,
                        })

DFN.add_fracture_family(shape="rect",
                        distribution="tpl",
                        alpha=1.31,
                        min_radius=1,
                        max_radius=5,
                        kappa=4.72,
                        plunge=51,
                        trend=263.0,
                        aspect=2,
                        p32=3.15e-01,
                        hy_variable='aperture',
                        hy_function='constant',
                        hy_params={
                            "mu": 3.67e-6,
                        })



DFN.print_domain_parameters()
DFN.make_working_directory(delete = True)
DFN.check_input()
DFN.create_network()
DFN.output_report()
DFN.visual_mode = True
DFN.mesh_network()
G = DFN.create_graph("intersection", "left", "right")
DFN.dump_json_graph(G, "carbonate_x1")
# DFN.plot_graph(G)