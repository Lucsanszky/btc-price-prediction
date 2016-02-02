var http      = require('http');
var Pusher    = require('pusher-client');
var express   = require('express');
var AWS       = require("aws-sdk");

var pusher         = new Pusher('de504dc5763aeef9ff52');
var trades_channel = pusher.subscribe('diff_order_book');

AWS.config.loadFromPath('/home/ec2-user/.ec2/credentials.json');

var docClient = new AWS.DynamoDB.DocumentClient();

var table = "OrderBook";

var server = http.createServer(
  function(request, response) {
    response.writeHead( 200, {"content-type": "text/plain"} );
    response.write("Order book data are written to the console and the DB...\n");
    response.end();
  }
);

server.listen(8080);

function getTimeStamp() {
	var currentdate = new Date(); 
	var datetime = currentdate.getDate() + "/"
                   + (currentdate.getMonth()+1)  + "/" 
                   + currentdate.getFullYear() + " "  
                   + currentdate.getHours() + ":"  
                   + currentdate.getMinutes() + ":" 
                   + currentdate.getSeconds();

    return datetime;
}

trades_channel.bind('trade', function(data) {
	var params = {
    	TableName: table,
    	Item: {
        	"Date": getTimeStamp(),
        	"OrderBookData": {
            	"bids": data['bids'],
            	"asks": data['asks'],
        	}
    	}
	};

    console.log("Adding a new item...");
	docClient.put(params, function(err, data) {
    	if (err) {
        	console.error("Unable to add item. Error JSON:", JSON.stringify(err, null, 2));
    	} else {
        	console.log("Added item:", JSON.stringify(params, null, 2));
    	}
	});
});
