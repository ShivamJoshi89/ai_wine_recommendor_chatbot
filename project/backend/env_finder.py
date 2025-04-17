import pymongo

client = pymongo.MongoClient(
    "mongodb+srv://shivamjoshi89us:DEuNYRPbElRfml0e@cluster0.twkupzt.mongodb.net/WineRecommendationProject?retryWrites=true&w=majority&appName=Cluster0",
    tls=True,
    tlsAllowInvalidCertificates=True  # Use for debugging only!
)
