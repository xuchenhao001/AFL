'use strict';

let express = require('express');
let router = express.Router();

let log4js = require('log4js');
let logger = log4js.getLogger('REST');
logger.level = 'DEBUG';

const { Gateway, Wallets } = require('fabric-network');
const fs = require('fs');
const path = require('path');

let common = require('./common');

let pickOrg=1

router.post('/invoke/:channelName/:chaincodeName', async function (req, res) {
  let channelName = req.params.channelName;
  let chaincodeName = req.params.chaincodeName;
  let body = req.body;

  logger.info("Received request type: " + body.message)
  await invoke(res, channelName, chaincodeName, body.message, JSON.stringify(req.body));
});


router.get('/test/echo', async function (req, res) {
  common.responseSuccess(res, req.body);
});

let query = async function(res, channelName, chaincodeName, queryFuncName, args) {
  let resultBuff = await submitRequest(channelName, chaincodeName, queryFuncName, args, false);
  let result = resultBuff.toString();
  logger.debug('Query result: ' + result)
  if (result) {
    let detail = JSON.parse(result);
    common.responseSuccess(res, detail);
  } else {
    common.responseSuccess(res, []);
  }
}

let invoke = async function(res, channelName, chaincodeName, invokeFuncName, args) {
  let maxTries = 30;
  let errMessage;
  while (maxTries>0) {
    try {
      await submitRequest(channelName, chaincodeName, invokeFuncName, args, false);
      common.responseSuccess(res, {});
      return;
    } catch (error) {
      errMessage = 'Failed to submit transaction: ' + error;
      // if dirty read happened, retry for 3 times
      // if (errMessage.indexOf('READ_CONFLICT') !== -1 || errMessage.indexOf('ENDORSEMENT_POLICY_FAILURE') !== -1) {
      //   maxTries--;
      // } else {
      //   maxTries=0;
      // }
      maxTries=0; // no retry now
    }
  }
  common.responseInternalError(res, errMessage);
};

let submitRequest = async function (channelName, chaincodeName, funcName, args, isQuery) {
  // pick an org from ccp first
  let orgName = pickOrgByCCP();
  // load the network configuration
  const ccpPath = path.resolve(__dirname, 'wallet', 'connection-' + orgName + '.json');
  const fileExists = fs.existsSync(ccpPath);
  if (!fileExists) {
    throw new Error(`no such file or directory: ${ccpPath}`);
  }
  let ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));

  // Create a new file system based wallet for managing identities.
  const walletPath = path.join(__dirname, 'wallet');
  const wallet = await Wallets.newFileSystemWallet(walletPath);
  logger.debug(`Wallet path: ${walletPath}`);

  // Check to see if we've already enrolled the user.
  let identity = await wallet.get(orgName);
  if (!identity) {
    logger.error('An identity for the user "' + orgName + '" does not exist in the wallet!!!!!');
    return;
  }

  // Create a new gateway for connecting to our peer node.
  const gateway = new Gateway();
  await gateway.connect(ccp, { wallet, identity: orgName, discovery: { enabled: true, asLocalhost: false } });

  // Get the network (channel) our contract is deployed to.
  const network = await gateway.getNetwork(channelName);

  // Get the contract from the network.
  const contract = network.getContract(chaincodeName);

  // Submit the specified transaction.
  let result;
  if (isQuery) {
    result = await contract.evaluateTransaction(funcName, args);
  } else {
    result = await contract.submitTransaction(funcName, args);
  }
  logger.info('Transaction has been submitted: ' + funcName);

  // Disconnect from the gateway.
  await gateway.disconnect();
  return result;
}

// pick org from org1 to the last org
let pickOrgByCCP = function () {
  const ccpPath = path.resolve(__dirname, 'wallet');
  let files = fs.readdirSync(ccpPath);
  let lastFileName = files.sort(function(a, b) {
    return a.length - b.length || // sort by length, if equal then
        a.localeCompare(b);    // sort by dictionary order
  }).pop();
  let lastOrg = lastFileName
      .replace(/^connection-org/, '')
      .replace(/\.json$/, '');
  let lastOrgNum = parseInt(lastOrg);
  if (pickOrg >= lastOrgNum) {
    pickOrg = 1;
  } else {
    pickOrg++;
  }
  let orgName = "org" + pickOrg;
  logger.debug("Pick [" + orgName + "] this time!");
  return orgName;
}

module.exports = router;
