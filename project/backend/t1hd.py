import ssl
from pymongo import MongoClient

# Create a default SSL context if needed (optional customization)
# If you need to customize certificate validation you can do it by using tlsCAFile, etc.
# NOTE: You do not pass this context via the MongoClient constructor anymore.
default_context = ssl.create_default_context()

# (Optional) If you want to force TLS1.2 or higher, you can try:
default_context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1  # Deprecation warnings may occur

# Use MongoClient without passing the ssl_context
client = MongoClient(
    "mongodb+srv://shivamjoshi89us:DEuNYRPbElRfml0e@cluster0.twkupzt.mongodb.net/WineRecommendationProject"
    "?retryWrites=true&w=majority&appName=Cluster0",
    tls=True,  # same as ssl=True
    # tlsAllowInvalidCertificates=False  # include if needed; default is to require valid certs
)

# Now you can use `client` to interact with your MongoDB Atlas cluster.
