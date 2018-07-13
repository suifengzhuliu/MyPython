# -*- coding:utf-8 -*-  
import json
import urllib2
date = "20170530"
server_url = "http://www.easybots.cn/api/holiday.php?d="
 
vop_url_request = urllib2.Request(server_url+date)
vop_response = urllib2.urlopen(vop_url_request)
 
vop_data= json.loads(vop_response.read())
 
print(vop_data)
 
if vop_data[date]=='0':
    print ("this day is weekday")
elif vop_data[date]=='1':
    print ('This day is weekend')
elif vop_data[date]=='2':
    print ('This day is holiday')
else:
    print( 'error')