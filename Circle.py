# import cv2,numpy as np , matplotlib.pyplot as plt
# img = cv2.imread('C:/Users/Amin/Desktop/flower.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

# blur = cv2.medianBlur(gray, 5)
# circle = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2,minDist=30,      
#                           param1=50,param2=30,minRadius=10,maxRadius=100)

# output = img.copy()
# if circle is not None:
#     circle = np.uint16(np.around(circle))
#     for x,y,r in circle[0, :]:
#         cv2.circle(output,(x,y),r,(0,255,0),2)
#         cv2.circle(output,(x,y),2,(0,0,255),3)

# plt.imshow(cv2.cvtColor(output,cv2.COLOR_BGR2RGB))
# plt.title('circle')
# plt.axis('off')
# plt.show()

# import cv2,numpy,matplotlib.pyplot as plt
# image = cv2.imread('C:/Users/Amin/Desktop/flower.jpg')
# gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# circle = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=1.2,minDist=30,param1=50,param2=30,minRadius=10,maxRadius=30)

# output = image.copy()
# if image is not None:
#     circle = numpy.uint16(numpy.around(circle))
#     for x,y,r in circle[0,:]:
#         cv2.circle(output,(x,y),r,(0,255,0),2)
#         cv2.circle(output,(x,y),3,(0,0,255),3)

# plt.imshow(cv2.cvtColor(output,cv2.COLOR_BGR2RGB))
# plt.title("circle")
# plt.axis('off')
# plt.show()        

# from langchain_neo4j import Neo4jGraph
# from dotenv import load_dotenv
# import os

# load_dotenv()
# NEO4J_URI = os.getenv("NEO4J_URI_LOCAL")
# NEO4J_USERNAME = os.getenv("NO4J_USERNAME")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD_LOCAL")

# graph = Neo4jGraph(
#     url = NEO4J_URI,
#     username = NEO4J_USERNAME,
#     password = NEO4J_PASSWORD
# )

# schema = graph.schema
# print(schema)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ۱. بارگذاری مدل
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ۲. تبدیل جملات به بردار (همه با یک مدل!)
texts = [
    "گربه روی میز خوابیده است",      # جمله ۱
    "یک پشمالو روی میز استراحت می‌کند", # جمله ۲ (مشابه ۱)
    "قیمت بیت‌کوین امروز بالا رفت"     # جمله ۳ (بی‌ربط)
]

embeddings = model.encode(texts)

# ۳. محاسبه شباهت کسینوسی
similarities = cosine_similarity(embeddings)

print("ماتریس شباهت:")
print(np.round(similarities, 2))

# خروجی مورد انتظار:
# [[1.00  0.87  0.15]   ← جمله ۱ با ۲ بسیار مشابه (۰.۸۷)
#  [0.87  1.00  0.12]   ← جمله ۱ با ۳ بی‌ربط (۰.۱۵)
#  [0.15  0.12  1.00]]