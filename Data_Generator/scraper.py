import string
import scrapy
from pydispatch import dispatcher
from scrapy import signals
import urllib.request
import pandas as pd
import re
import csv
from urllib.request import urlretrieve


class ScrapperSpider(scrapy.Spider):
    baseUrl = "http://platesmania.com/fr/gallery-"
    name = 'licenceScraper'
    allowed_domains = ['platesmania.com']
    start_urls = [
        "http://platesmania.com/fr/gallery-1"
    ]
    csvFilepath = './image/dataScraped.csv'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    page = 1
    maxPage = 50
    delimiter=','
    DIRECTORY_IMG_PLATE = './image/plate/'
    DIRECTORY_IMG_GLOBAL = './image/car/'


    GLOBAL_DATA = None

    def __init__(self):
        dispatcher.connect(self.end, signals.spider_closed)
        dispatcher.connect(self.start, signals.spider_opened)

    def start(self, spider):
        pass

    def end(self, spider):
        pass
    
    def parse(self, response):
        response.headers=self.headers
        # J'utilise des selecteurs HTML pour atteindre mes images souhaités sur le site distant
        # Mais vous pouvez utiliser aussi des selecteurs CSS
        # Je récupère ici 6 images par page
        imgContenerAll = response.xpath('.//div[@class="panel panel-grey"]')

        # Pour mes 6 images par page :
        for imgContener in imgContenerAll:
            # J'utilise de nouveaux selecteur
            panelBody = imgContener.xpath('div[@class="panel-body"]')
            
            # Je récupère les champs de texte voulu (attribut text())
            carType = panelBody.xpath('.//h4/a/text()').get().split(' ')

            voitureMarque = carType[0]
            voitureModele = carType[1]


            subContenerImgGlobal = imgContener.xpath('.//div[@class="row"]')[1]
            subContenerImgPlate = imgContener.xpath('.//div[@class="row"]')[2]

            # Je récupère les urls vers les images (via attribut @src)
            urlImgGlobal = subContenerImgGlobal.xpath('.//a//img/@src').get()
            urlImgPlate = subContenerImgPlate.xpath('.//a//img/@src').get()
            plateNumber = subContenerImgPlate.xpath('.//a//img/@alt').get()

            imgGlobalName = urlImgGlobal.split('/')[-1]
            imgPlateName = urlImgPlate.split('/')[-1]

            # Déstination ou sauvegarder nos images, on prends les folder de base et on ajout le nom de l'image
            destinationFolderImgPlate = self.DIRECTORY_IMG_PLATE + (re.sub(r'\([^)]*\)', '',plateNumber)).replace(" ","-")+".png"
            destinationFolderImgGlobal = self.DIRECTORY_IMG_GLOBAL + (re.sub(r'\([^)]*\)', '',plateNumber)).replace(" ","-")+".png"
            
            # On créer un n-uplet de donnée, selon notre image en cours de scraping
            """    'voitureMarque'
                'voitureModele'
                'imgGlobalName'
                'imgPlaqueName'
                'plateNumber'"""
            row = [
                voitureMarque,
                voitureModele,
                imgGlobalName,
                imgPlateName,
                plateNumber
            ]

            # On verifie que celui-ci n'existe pas dans notre fichier CSV, on évite les doublons
            with open(self.csvFilepath, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
            
            req1 = urllib.request.Request(urlImgGlobal, headers=self.headers)
            with urllib.request.urlopen(req1) as response, open(destinationFolderImgGlobal, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            req2 = urllib.request.Request(urlImgPlate, headers=self.headers)
            with urllib.request.urlopen(req2) as response, open(destinationFolderImgPlate, 'wb') as out_file:
                data = response.read()
                out_file.write(data)

        # Incrémentation de la page pour scrap la suivante
        self.page += 1
        
        # On demande à Scrapy d'aller scrap la page suivante souhaité
        if self.page < self.maxPage:
            yield scrapy.Request(url=self.baseUrl+str(self.page), callback=self.parse)


