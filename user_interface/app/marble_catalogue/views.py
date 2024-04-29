import json
import os
from rest_framework.views import APIView
from django.http import HttpResponse
from .utils import get_images, get_base64
from django.shortcuts import render
from django.conf import settings


image_path_to_caption_file = os.path.join(
    settings.STATIC_ROOT, 'image_path_to_caption.json')
images_dir = os.path.join(settings.STATIC_ROOT, 'images')
with open(image_path_to_caption_file, 'r') as f:
    image_path_to_caption = json.load(f)
img_paths = list(image_path_to_caption.keys())
desc = list(image_path_to_caption.values())


class Marbles(APIView):

    def get(self, request, search_query):

        retrieval_list = get_images(search_query, 16)
        result = [
            {
                'img_rank': index,
                'img_data': get_base64(os.path.join(images_dir, img_paths[i])),
                'img_path':img_paths[i],
                'img_desc':' '.join([str(j) for j in desc[i]])
            }
            for index, i in enumerate(retrieval_list)]

        return HttpResponse(json.dumps(result), content_type="application/json")


def index(request):
    return render(request, 'index.html')
