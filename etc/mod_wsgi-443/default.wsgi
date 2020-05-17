
CONTENT = b'''
<html>
<head>
<title>My web site runs on Malt Whiskey</title>
</head>
<body style="margin-top: 100px;">
<table align="center"; style="width: 850px;" border="0" cellpadding="30">
<tbody>
<tr>
<td>
<img style="width: 275px; height: 445px;"
  src="/__wsgi__/images/snake-whiskey.jpg">
</td>
<td style="text-align: center;">
<span style="font-family: Arial,Helvetica,sans-serif;
  font-weight: bold; font-size: 70px;">
My web site<br>runs on<br>Malt Whiskey<br>
<br>
</span>
<span style="font-family: Arial,Helvetica,sans-serif;
  font-weight: bold;">
For further information on configuring mod_wsgi,<br>
see the <a href="http://www.modwsgi.org/">documentation</a>.
</span>
</td>
</tr>
</tbody>
</table>
</body>
</html>
'''

def application(environ, start_response):
    status = '200 OK'
    output = CONTENT

    response_headers = [('Content-type', 'text/html'),
                        ('Content-Length', str(len(output)))]
    start_response(status, response_headers)

    return [output]

