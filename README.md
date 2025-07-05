<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Facial Emotion Recognition with Streamlit</title>
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      max-width: 900px;
      margin: auto;
      padding: 2rem;
      background: #fdfdfd;
      line-height: 1.7;
    }
    h1, h2 {
      color: #2c3e50;
    }
    pre {
      background: #f4f4f4;
      padding: 1rem;
      border-left: 5px solid #3498db;
      overflow-x: auto;
    }
    code {
      background: #eee;
      padding: 2px 6px;
      border-radius: 4px;
    }
    ul {
      list-style: square;
      margin-left: 1.5rem;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      margin-top: 1rem;
    }
    th, td {
      border: 1px solid #ccc;
      padding: 0.6rem;
      text-align: left;
    }
    a {
      color: #3498db;
    }
  </style>
</head>
<body>

  <h1>😃 Facial Emotion Recognition with Streamlit</h1>

  <p>This project detects facial emotions using a fine-tuned ResNet18 model. It’s integrated with a Streamlit app that allows users to upload images and view the top 3 predicted emotions with confidence scores.</p>

  <h2>📸 Demo</h2>
  <p><em>(Insert a screenshot or demo gif here)</em></p>

  <h2>💡 Key Features</h2>
  <ul>
    <li>Real-time facial emotion classification</li>
    <li>Powered by ResNet18 via transfer learning</li>
    <li>Trained on the FER-2013 dataset</li>
    <li>Streamlit web app for interactive testing</li>
    <li>Supports 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral</li>
  </ul>

  <h2>🧠 Model</h2>
  <p>Uses pretrained ResNet18 from PyTorch with modified output layer for 7 classes. Includes:</p>
  <ul>
    <li>Data augmentation</li>
    <li>Dropout (0.4)</li>
    <li>StepLR learning rate scheduling</li>
  </ul>

  <h2>📂 Project Structure</h2>
  <pre><code>
Facial-Emotion-Recognition-with-Streamlit/
├── app.py
├── improved_emotion_resnet18.py
├── models/
│   └── emotion_resnet18.py
├── improved_emotion_resnet18.pth
├── requirements.txt
├── README.md
└── dataset/
    ├── train/
    └── val/
  </code></pre>

  <h2>🚀 Run the Web App</h2>
  <pre><code>streamlit run app.py</code></pre>

  <h2>📈 Train the Model</h2>
  <pre><code>python improved_emotion_resnet18.py</code></pre>

  <h2>🔍 Results</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Train Accuracy</td><td>81.93%</td></tr>
    <tr><td>Validation Accuracy</td><td>62.78%</td></tr>
    <tr><td>Model</td><td>ResNet18</td></tr>
    <tr><td>Best Model Saved</td><td>✅ Yes (on val acc improvement)</td></tr>
  </table>

  <h2>📚 References</h2>
  <ul>
    <li><a href="https://www.kaggle.com/datasets/msambare/fer2013" target="_blank">FER-2013 Dataset</a></li>
    <li><a href="https://pytorch.org/vision/stable/models.html" target="_blank">PyTorch ResNet Docs</a></li>
    <li><a href="https://streamlit.io/" target="_blank">Streamlit</a></li>
  </ul>

  <h2>🤝 Contributing</h2>
  <p>Pull requests are welcome! For major changes, open an issue to discuss improvements first.</p>

  <h2>📝 License</h2>
  <p>This project is licensed under the <strong>MIT License</strong>.<br/>
     You are free to use, modify, and distribute this software with proper credit.
  </p>

</body>
</html>
