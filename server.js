var express = require(‘express’);
var app = express();
app.set(‘port’, process.env.PORT || 1234);
app.listen(app.get(‘port’));
console.log('hello world');
