var http      = require('http');
var Pusher    = require('pusher-client');
var mongoose  = require('mongoose');
var express   = require('express');

var pusher         = new Pusher('de504dc5763aeef9ff52');
var trades_channel = pusher.subscribe('live_trades');
var app            = express();

var config = {
      "USER"     : "",                  // if your database has user/pwd defined
      "PASS"     : "",
      "HOST"     : "ip-10-0-2-63.ec2.internal",  // the domain name of our MongoDB EC2 instance
      "PORT"     : "27017",             // this is the default port mongoDB is listening for incoming queries
      "DATABASE" : "test"         // the name of your database on that instance
    };

var dbPath  = "mongodb://" + config.USER + ":" +
    config.PASS + "@"+
    config.HOST + ":"+
    config.PORT + "/"+
    config.DATABASE;

var standardGreeting = 'Hello World!';

var db;              // our MongoDb database

var greetingSchema;  // our mongoose Schema
var Greeting;        // our mongoose Model

// create our schema
greetingSchema = mongoose.Schema({
  sentence: String
});
// create our model using this schema
Greeting = mongoose.model('Greeting', greetingSchema);

// ------------------------------------------------------------------------
// Connect to our Mongo Database hosted on another server
//
console.log('\nattempting to connect to remote MongoDB instance on another EC2 server '+config.HOST);

if ( !(db = mongoose.connect(dbPath)) )
  console.log('Unable to connect to MongoDB at '+dbPath);
else 
  console.log('connecting to MongoDB at '+dbPath);

// connection failed event handler
mongoose.connection.on('error', function(err){
  console.log('database connect error '+err);
}); // mongoose.connection.on()

var server = http.createServer(
  function(request, response) {
    response.writeHead( 200, {"content-type": "text/plain"} );
    response.write("Trades are written to the console...\n");
    response.end();
  }
);

server.listen( 8080 );

trades_channel.bind('trade', function(data) {
  console.log(data);
});
