import gradio as gr
import numpy as np
import torch
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn

# Định nghĩa lại SimpleNN và ProdLDA
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.output(x)
        return x

class ProdLDA(nn.Module):
    def __init__(self, vocab_size, num_topics=50, en_units=200, dropout=0.4):
        super().__init__()
        self.num_topics = num_topics
        import numpy as np
        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))
        self.mu2.requires_grad = False
        self.var2.requires_grad = False
        self.fc11 = nn.Linear(vocab_size, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)
        self.mean_bn = nn.BatchNorm1d(num_topics, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn.weight.data.copy_(torch.ones(num_topics))
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topics, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn.weight.data.copy_(torch.ones(num_topics))
        self.logvar_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(vocab_size, eps=0.001, momentum=0.001, affine=True)
        self.decoder_bn.weight.data.copy_(torch.ones(vocab_size))
        self.decoder_bn.weight.requires_grad = False
        self.fc1_drop = nn.Dropout(dropout)
        self.theta_drop = nn.Dropout(dropout)
        self.fcd1 = nn.Linear(num_topics, vocab_size, bias=False)
        nn.init.xavier_uniform_(self.fcd1.weight)
    def get_theta(self, x):
        mu, logvar = self.encode(x)
        import torch.nn.functional as F
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)
        theta = self.theta_drop(theta)
        if self.training:
            return theta, mu, logvar
        else:
            return theta
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu
    def encode(self, x):
        import torch.nn.functional as F
        e1 = F.softplus(self.fc11(x))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_drop(e1)
        return self.mean_bn(self.fc21(e1)), self.logvar_bn(self.fc22(e1))
    def forward(self, x):
        theta, mu, logvar = self.get_theta(x)
        return theta

label_names = ["World", "Sports", "Business", "Sci/Tech"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load các mô hình và vectorizer đã lưu
nn_ckpt = torch.load('param/nn_model.pth', map_location=device, weights_only=False)
nn_vectorizer = nn_ckpt['vectorizer']
nn_label_encoder = nn_ckpt['label_encoder']
input_dim = len(nn_vectorizer.get_feature_names_out())
hidden_dim = 128
output_dim = len(nn_label_encoder.classes_)
nn_model = SimpleNN(input_dim, hidden_dim, output_dim)
nn_model.load_state_dict(nn_ckpt['model_state_dict'])
nn_model = nn_model.to(device)
nn_model.eval()

nb_model = joblib.load('param/nb_model.pkl')
nb_vectorizer = joblib.load('param/nb_vectorizer.pkl')
nb_label_encoder = joblib.load('param/nb_label_encoder.pkl')

lr_model = joblib.load('param/lr_model.pkl')
lr_vectorizer = joblib.load('param/lr_vectorizer.pkl')
lr_label_encoder = joblib.load('param/lr_label_encoder.pkl')

svc_model = joblib.load('param/svc_model.pkl')
svc_vectorizer = joblib.load('param/svc_vectorizer.pkl')
svc_label_encoder = joblib.load('param/svc_label_encoder.pkl')

sgd_model = joblib.load('param/sgd_model.pkl')
sgd_vectorizer = joblib.load('param/sgd_vectorizer.pkl')
sgd_label_encoder = joblib.load('param/sgd_label_encoder.pkl')

rf_model = joblib.load('param/rf_model.pkl')
rf_vectorizer = joblib.load('param/rf_vectorizer.pkl')
rf_label_encoder = joblib.load('param/rf_label_encoder.pkl')

prodlda_vectorizer = joblib.load('param/prodlda_bow_vectorizer.pkl')
prodlda_svm = joblib.load('param/prodlda_svm.pkl')
prodlda_model = ProdLDA(vocab_size=len(prodlda_vectorizer.vocabulary_), num_topics=50)
prodlda_model.load_state_dict(torch.load('param/prodlda.pth', map_location=device))
prodlda_model = prodlda_model.to(device)
prodlda_model.eval()

# Load pre-computed centroids
centroids = joblib.load('param/cluster_centroids.pkl')
centroids_matrix = np.asarray(np.vstack(centroids))

def cluster_infer(text):
    vec = nb_vectorizer.transform([text])
    similarities = cosine_similarity(vec, centroids_matrix)
    pred = np.argmax(similarities, axis=1)[0]
    return label_names[int(pred)]

def prodlda_infer(text):
    bow = prodlda_vectorizer.transform([text])
    input_tensor = torch.FloatTensor(bow.toarray()).to(device)
    with torch.no_grad():
        theta = prodlda_model.get_theta(input_tensor)
        if isinstance(theta, tuple):
            theta = theta[0]
        pred = prodlda_svm.predict(theta.cpu().numpy())[0]
    return label_names[int(pred)]

def infer(model_name, input_text):
    if model_name == "Neural Network":
        vec = nn_vectorizer.transform([input_text])
        input_tensor = torch.FloatTensor(vec.toarray()).to(device)
        with torch.no_grad():
            outputs = nn_model(input_tensor)
            _, pred = torch.max(outputs, 1)
            pred = pred.item()
        return label_names[int(pred)]
    elif model_name == "Naive Bayes":
        vec = nb_vectorizer.transform([input_text])
        pred = nb_model.predict(vec)[0]
        return label_names[int(pred)]
    elif model_name == "Logistic Regression":
        vec = lr_vectorizer.transform([input_text])
        pred = lr_model.predict(vec)[0]
        return label_names[int(pred)]
    elif model_name == "SVM":
        vec = svc_vectorizer.transform([input_text])
        pred = svc_model.predict(vec)[0]
        return label_names[int(pred)]
    elif model_name == "SGD":
        vec = sgd_vectorizer.transform([input_text])
        pred = sgd_model.predict(vec)[0]
        return label_names[int(pred)]
    elif model_name == "Random Forest":
        vec = rf_vectorizer.transform([input_text])
        pred = rf_model.predict(vec)[0]
        return label_names[int(pred)]
    elif model_name == "Prod LDA":
        return prodlda_infer(input_text)
    elif model_name == "Cluster":
        return cluster_infer(input_text)
    else:
        return "Unknown model"

model_choices = [
    "Neural Network", "Naive Bayes", "Logistic Regression", "SVM", "SGD", "Random Forest", "Prod LDA", "Cluster"
]

if __name__ == "__main__":
    gr.Interface(
        fn=infer,
        inputs=[
            gr.Dropdown(choices=model_choices, label="Select Model"),
            gr.Textbox(lines=4, label="Enter News Content")
        ],
        outputs=gr.Label(label="Predicted Category"),
        title="AG News Text Classification",
        description="Choose a model and enter a news article to classify it into one of 4 categories: World, Sports, Business, Sci/Tech."
    ).launch(server_name="0.0.0.0", server_port=7860) 