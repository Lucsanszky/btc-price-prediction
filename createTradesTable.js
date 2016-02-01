var AWS = require("aws-sdk");

AWS.config.loadFromPath('~/.ec2/credentials.json');

var dynamodb = new AWS.DynamoDB();

var params = {
    TableName : "Trades",
    KeySchema: [       
        { AttributeName: "TradeID", KeyType: "HASH"},  //Partition key
        { AttributeName: "Date", KeyType: "RANGE" }  //Sort key
    ],
    AttributeDefinitions: [       
        { AttributeName: "TradeID", AttributeType: "N" },
        { AttributeName: "Date", AttributeType: "S" }
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