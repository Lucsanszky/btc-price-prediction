var http      = require('http');
var Pusher    = require('pusher-client');
var express   = require('express');
var AWS       = require("aws-sdk");

var pusher         = new Pusher('de504dc5763aeef9ff52');
var trades_channel = pusher.subscribe('live_trades');

AWS.config.update({
  endpoint: "https://dynamodb.us-west-2.amazonaws.com"
});

var dynamodb = new AWS.DynamoDB();

var params = {
    TableName : "Trades",
    KeySchema: [       
        { AttributeName: "TradeID", KeyType: "HASH"},  //Partition key
        { AttributeName: "Date", KeyType: "RANGE" }  //Sort key
    ],
    AttributeDefinitions: [       
        { AttributeName: "TradeID", AttributeType: "N" },
        { AttributeName: "Date", AttributeType: "N" }
    ],
    ProvisionedThroughput: {       
        ReadCapacityUnits: 5, 
        WriteCapacityUnits: 5
    }
};

dynamodb.createTable(params, function(err, data) {
    if (err) {
        console.error("Unable to create table. Error JSON:", JSON.stringify(err, null, 2));
    } else {
        console.log("Created table. Table description JSON:", JSON.stringify(data, null, 2));
    }
});

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
