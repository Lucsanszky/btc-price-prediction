var express = require(‘express’);
var app = express();
app.set(‘port’, process.env.PORT || 8081);
app.listen(app.get(‘port’));
console.log('hello world');
