## ⚙️ Getting Started

1. Download Resources
   * Download `RiskBench_Dataset` [here](https://nycu1-my.sharepoint.com/:f:/g/personal/ychen_m365_nycu_edu_tw/EviA5ovlh6hPo_ZXEPQjxAQB2R3vNubk3HM1u4ib1VdPFA?e=WHEWdm).
   <!-- * Download `metadata.zip` [here](https://nycu1-my.sharepoint.com/personal/ychen_m365_nycu_edu_tw/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fychen%5Fm365%5Fnycu%5Fedu%5Ftw%2FDocuments%2FRiskBench%2FDATA%5FFOR%5FPlanning%5FAware%5FMetric). -->

2. Extract Files
   * `RiskBench_Dataset`
   * `metadata.zip`
   * `model.zip`
3. Install Dependencies

	Install the required dependencies in your preferred environment. You can start by setting up a new Conda environment:
	```bash
	conda create -n analysis_tool python=3.7
	conda activate analysis_tool
	cd ${TOOL_ROOT}

	conda  install pyqt
	pip3 install -r requirements.txt
	```

## 🔍 Quantitative Results For Risk Object Identification (ROI)


Execute the following command to perform risk object identification:

```bash
python ROI_tool.py --method ${MODEL} --metadata_root ${METADATA_ROOT} --save_result --result_path ${ROI_PATH}
```

The output results will be saved to `${ROI_PATH}/${MODEL}/${DATA_TYPE}.josn`


## 📊 Fine-grained Scenario-based Analysis

1. Run the Visualization Tool
	
	Execute the command below for scenario-based analysis:
	```bash
	python ROI_vis_tool.py --data_root ${DATASET_ROOT} --metadata_root ${METADATA_ROOT} --vis_result_path ${VIS_PATH}
	```
2. Choose the model and scenario type.
3. Select the attributes of interest.
4. Click Filter Scenario.
5. Select a specific scenario for analysis:
    * Click Generate Video to save the result as a GIF.
    * Click Generate JSON to save the quantitative results in a JSON file.

	The generated results will be saved in the following directories:
   * GIF : `${VIS_PATH}/gif/${MODEL}/${DATA_TYPE}`
   * JSON : `{VIS_PATH}/json/${MODEL}/${DATA_TYPE}`


![Fine-grained Scenario-based Analysis](./utils/demo-ROI.gif)

