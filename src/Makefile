
PYTHONC = ../../miniconda3/envs/carnd-term1/bin/python
PYTHONG = LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64 ../../miniconda3/envs/carnd-term1-gpu/bin/python

UNAME = ${shell uname}
ifeq ($(UNAME),Darwin)
PYTHONC = ../../../../src/miniconda3/envs/carnd-term1/bin/python
PYTHONG = ../../../../src/miniconda3/envs/carnd-term1/bin/python
endif


all: 
	${PYTHONC} lesson37_MultipleDetections.py
	#${PYTHONC} lesson34_SearchandClassify_NotWork.py
	#${PYTHONC} lesson34_SearchandClassify_NotWork_load.py
	#${PYTHONC} lesson35_HogSub-samplingWindowSearch.py
	#${PYTHONC} lesson28_ColorClassify.py
	#${PYTHONC} lesson22_CombineAndNormalizeFeatures.py
	#${PYTHONC} lesson20_scikit-HOG.py
	#${PYTHONC} lesson19_DataExploration.py
	#${PYTHONC} lesson16_SpatialBinnigColor.py
	#${PYTHONC} lesson15_ColorSpaces.py

run: 
	#${PYTHONC} project_training.py
	${PYTHONC} project_pipeline.py

video:
	${PYTHONC} project_pipeline_video.py

calib:
	${PYTHONC} project_code01.py

test:
#	${PYTHONC} test02.py
#	${PYTHONC} test01.py

