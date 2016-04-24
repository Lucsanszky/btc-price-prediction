/*
** Source: http://stackoverflow.com/questions/15543418/how-to-get-the-result-of-describetable-amazon-dynamodb-method-in-node-js
*/

var AWS = require("aws-sdk");
var async = require('async');

AWS.config.loadFromPath('/home/ec2-user/.ec2/credentials.json');

var docClient = new AWS.DynamoDB.DocumentClient();

var params = {
    TableName: 'OrderBook',
    Count: 'true'
};

var scanComplete = false,
    itemCountTotal = 0,
    consumedCapacityUnitsTotal = 0;

async.until( function() { return scanComplete; },
             function (callback) {
                docClient.scan(params, function (err, result) {
                    if (err) {
                        console.log(err);
                    } else {
                        console.log(result);
                        
                        if (typeof (result.LastEvaluatedKey) === 'undefined' ) {
                            scanComplete = true;
                        } else {
                            params.ExclusiveStartKey = result.LastEvaluatedKey;
                        }

                        itemCountTotal += result.Count;
                        consumedCapacityUnitsTotal += result.ConsumedCapacityUnits;

                        if (!scanComplete) {
                            console.log("cumulative itemCount " + itemCountTotal);
                            console.log("cumulative capacity units " + consumedCapacityUnitsTotal);
                        }
                    }
                    callback(err);
                });
             },

             function (err) {
                if (err) {
                    console.log('error in processing scan ');
                    console.log(err);
                } else {
                    console.log('scan complete')
                    console.log('Total items: ' + itemCountTotal);
                    console.log('Total capacity units consumed: ' + consumedCapacityUnitsTotal);
                }
             }
);
