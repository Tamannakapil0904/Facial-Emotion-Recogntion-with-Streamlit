
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
