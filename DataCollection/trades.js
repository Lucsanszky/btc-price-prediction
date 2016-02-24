var http      = require('http');
var Pusher    = require('pusher-client');
var express   = require('express');
var AWS       = require("aws-sdk");

var pusher         = new Pusher('de504dc5763aeef9ff52');
var trades_channel = pusher.subscribe('live_trades');

AWS.config.loadFromPath('/home/ec2-user/.ec2/credentials.json');

var docClient = new AWS.DynamoDB.DocumentClient();

var table = "Trades";

var server = http.createServer(
  function(request, response) {
    response.writeHead( 200, {"content-type": "text/plain"} );
    response.write("Trades are written to the console and the DB...\n");
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
        	"TradeID": data['id'],
        	"Date": getTimeStamp(),
        	"TradeData": {
            	"amount": data['amount'],
            	"price": data['price'],
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
