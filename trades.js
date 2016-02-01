var http      = require('http');
var Pusher    = require('pusher-client');
var express   = require('express');
var AWS       = require("aws-sdk");

var pusher         = new Pusher('de504dc5763aeef9ff52');
var trades_channel = pusher.subscribe('live_trades');

var server = http.createServer(
  function(request, response) {
    response.writeHead( 200, {"content-type": "text/plain"} );
    response.write("Trades are written to the console...\n");
    response.end();
  }
);

server.listen( 8080 );

trades_channel.bind('trade', function(data) {
  //console.log(data);
});
