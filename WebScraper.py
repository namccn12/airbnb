
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import csv

reader = csv.reader(open('C:\\Users\\798018\\Documents\\SparkData\\Airbnb\\sampleids.csv'))

mylist = list(reader)

print(mylist)

idrange = mylist
for i in idrange:
    id = str(i[0])
    url = 'https://www.airbnb.com.au/rooms/'+id
    #print(url)


    response = requests.get(url)

    #print(response)

    soup = BeautifulSoup(response.text, "html.parser")
    # print(soup.prettify())
    # soup.findAll('a')
    # print(soup.findAll('span'))
    # print(soup.find_all('span'))
    rating_span = soup.find_all('span', class_='_1s5p755r')
    rating_text = str(rating_span)

    if rating_text != '[]':
        rating = rating_text[26:30]
        if "ou" in rating:
            rating = rating[0:1]
        print(id+"    " + rating)


    list_stuff =soup.find_all('div',class_="_czm8crp")
    for j in range(1,len(list_stuff)):
        print(list_stuff[j].text)
# for now I can't get these as they require the page to load.
# print("***location***")
# print(soup.find_all('span', class_='_15vir1vr'))
#
# print(soup)



