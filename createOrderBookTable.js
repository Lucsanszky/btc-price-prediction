var AWS = require("aws-sdk");

AWS.config.loadFromPath('/home/ec2-user/.ec2/credentials.json');

var dynamodb = new AWS.DynamoDB();

var params = {
    TableName : "OrderBook",
    KeySchema: [       
        { AttributeName: "Date", KeyType: "HASH"}  //Partition key
    ],
    AttributeDefinitions: [       
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