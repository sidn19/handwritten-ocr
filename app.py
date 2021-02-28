from bottle import get, run, view

@get('/')
@view('index')
def index():
    pass

run(host='localhost', port=8080, debug=True, reloader=True)