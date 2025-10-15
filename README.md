🧠 ML Challenge 2025 – Smart Product Pricing Challenge
📘 Problem Overview

In modern e-commerce, determining the optimal price point for products is essential for marketplace success and customer satisfaction.

Your challenge is to develop an ML model that analyzes product details — including text descriptions and images — to predict the price of each product.

The relationship between product attributes and pricing is complex. Factors such as brand, specifications, and Item Pack Quantity (IPQ) directly influence how a product is priced. Your model should capture these relationships holistically to suggest an optimal price.

📊 Data Description

The dataset includes detailed product information as follows:

Column	Description
sample_id	Unique identifier for each input sample
catalog_content	Text field containing product title, description, and Item Pack Quantity (IPQ) concatenated
image_link	Public URL to the product image (e.g. https://m.media-amazon.com/images/I/71XfHPR36-L.jpg)
price	Target variable (available only in the training data)
📁 Dataset Files
File	Description
dataset/train.csv	Training file with price labels
dataset/test.csv	Test file without price labels (use this for predictions)
dataset/sample_test.csv	Sample input for testing the pipeline
dataset/sample_test_out.csv	Sample output file format (use the same structure for submission)
🧩 Source Files
File	Description
src/utils.py	Helper functions for downloading product images using image_link. Includes retry logic for throttling.
src/test.ipynb	Example notebook showing how to use download_images function.
sample_code.py	Sample code to generate output in the required format (optional to use).
⚙️ Output Format

Your submission must be a CSV file with exactly the following two columns:

Column	Description
sample_id	Unique identifier matching the test record
price	Predicted float value of the product’s price
✅ Important:

Include all sample IDs from test.csv

Predicted prices must be positive float values

Output format must match sample_test_out.csv exactly

📈 Evaluation Metric

Submissions are evaluated using Symmetric Mean Absolute Percentage Error (SMAPE):

𝑆
𝑀
𝐴
𝑃
𝐸
=
1
𝑛
∑
∣
𝑝
𝑟
𝑒
𝑑
𝑖
𝑐
𝑡
𝑒
𝑑
_
𝑝
𝑟
𝑖
𝑐
𝑒
−
𝑎
𝑐
𝑡
𝑢
𝑎
𝑙
_
𝑝
𝑟
𝑖
𝑐
𝑒
∣
(
∣
𝑎
𝑐
𝑡
𝑢
𝑎
𝑙
_
𝑝
𝑟
𝑖
𝑐
𝑒
∣
+
∣
𝑝
𝑟
𝑒
𝑑
𝑖
𝑐
𝑡
𝑒
𝑑
_
𝑝
𝑟
𝑖
𝑐
𝑒
∣
)
/
2
SMAPE=
n
1
	​

∑
(∣actual_price∣+∣predicted_price∣)/2
∣predicted_price−actual_price∣
	​

Example:

If actual price = 100 and predicted = 120

𝑆
𝑀
𝐴
𝑃
𝐸
=
∣
100
−
120
∣
(
100
+
120
)
/
2
×
100
=
18.18
%
SMAPE=
(100+120)/2
∣100−120∣
	​

×100=18.18%
⚖️ Constraints

Predictions must be positive floats

Model must use a MIT or Apache 2.0 License

Final model size must be ≤ 8 Billion parameters

Submission file must exactly match the expected format — any mismatch will result in evaluation failure

🧮 Suggested Approach

Data Preparation

Parse and clean the catalog_content

Extract text embeddings using NLP models (e.g., Sentence-BERT)

Download and preprocess images (resize, normalize)

Extract image embeddings using CNNs (e.g., ResNet, EfficientNet)

Feature Fusion

Combine text and image embeddings

Include numerical features (e.g., IPQ if parsed)

Modeling

Use models such as LightGBM, XGBoost, or multimodal transformers

Tune hyperparameters with validation splits

Evaluation

Measure SMAPE on validation data

Ensure no missing predictions in the test output

Submission

Generate predictions for all test samples

Format as per sample_test_out.csv

🧰 Example Usage
python sample_code.py --train dataset/train.csv --test dataset/test.csv --output submission.csv

🏁 Deliverables

Trained model files

Inference script or notebook

Final output CSV (submission.csv)

README.md (this document)
