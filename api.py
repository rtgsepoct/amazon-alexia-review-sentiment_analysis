
from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO

# nltk.cropus import stopwords
from nltk.stem.porter import PorterStemmer