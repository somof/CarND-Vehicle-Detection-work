
PYTHONC = ../../miniconda3/envs/carnd-term1/bin/python
PYTHONG = LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64 ../../miniconda3/envs/carnd-term1-gpu/bin/python

UNAME = ${shell uname}
ifeq ($(UNAME),Darwin)
PYTHONC = ../../../../src/miniconda3/envs/carnd-term1/bin/python
PYTHONG = ../../../../src/miniconda3/envs/carnd-term1/bin/python
endif


all: 
	#${PYTHONC} project_training.py
	#${PYTHONC} project_search.py
	${PYTHONC} project_pipeline.py

lesson:
	#${PYTHONC} 00lesson/lesson37_MultipleDetections.py
	#${PYTHONC} 00lesson/lesson34_SearchandClassify_NotWork.py
	#${PYTHONC} 00lesson/lesson34_SearchandClassify_NotWork_load.py
	#${PYTHONC} 00lesson/lesson35_HogSub-samplingWindowSearch.py
	#${PYTHONC} 00lesson/lesson28_ColorClassify.py
	#${PYTHONC} 00lesson/lesson22_CombineAndNormalizeFeatures.py
	#${PYTHONC} 00lesson/lesson20_scikit-HOG.py
	#${PYTHONC} 00lesson/lesson19_DataExploration.py
	#${PYTHONC} 00lesson/lesson16_SpatialBinnigColor.py
	#${PYTHONC} 00lesson/lesson15_ColorSpaces.py

run: 
	${PYTHONC} project_training.py
	${PYTHONC} project_pipeline.py

video:
	${PYTHONC} project_pipeline_video.py
