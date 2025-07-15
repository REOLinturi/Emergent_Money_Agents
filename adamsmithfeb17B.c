#include <stdheaders.h>
#include <limits.h>
#define FILENAME "adamsmith.txt"
#define INTMULTIPLIER 10
#define POPULATION 300
#define MAXGIFTED 60
#define MAXGROUP 10
#define SKILLS 31
#define INITIALPRICE 1.0f
#define INITIALEFFICIENCY 1.0f
#define GIFTEDINITIALEFFICIENCY 2.0f
#define GIFTEDEFFICIENCYMINIMUM 1.5f
#define INITIALSOCIALTRANSPARENCY 0.70f
#define MAXSURPLUSFACTOR 2
#define MAXPERIODS 2000 //test purposes
#define MAXUSEFULINDEX 10000000
#define DISCONT 0.8f //periodically transactions, (1-DISCONT) recentlyproduced are forgotten
#define HISTORY 4 // must be 1 / (1-DISCONT) - 1 for the initial state to show continuous production level
#define SPOILEDSURPLUSDELAY (2 * HISTORY)
#define PRICEDEMANDELASTICITY 2 //0 -- no elasticity, 1 -- $ value stays the same, 2 -- $ value increases
#define BASICROUNDELASTIC 1 //True or false (1/0) false means only surplusrounds are run with elastic needs, true sets all rounds to elastic needs
#define SPOILSURPLUSEXCESS 0.10f
#define STOCKSPOILTRESHOLD 2.0f
#define PRICEREDUCTION 0.95f
#define PRICEHIKE 1.05f
#define PRICELEAP 1.3f
//#define SPENDINGFROMEXCESS 0.50f //needsincrement is reduced in each incremental round by SPENDINGFROMEXCESS %. Low value increases time used.
//#define MAXNEEDSINCREMENT 1.3f //If one becomes wealthy, needs increase, but not immediately proportionally
#define MAXNEEDSINCREASE 1.5f //If one becomes wealthy, needslevel increases, but not immediately proportionally
#define SMALLNEEDSINCREASE 1.05f //If one becomes less wealthy, needslevel is reduced when one tries to make ends meet
#define MAXNEEDSREDUCTION 0.7f //If one becomes less wealthy, needslevel is reduced when one tries to make ends meet
#define SMALLNEEDSREDUCTION 0.95f //If one becomes less wealthy, needslevel is reduced when one tries to make ends meet
#define MAXRISEINELASTICNEED 1.01f //Socially affected needs change slowly, marketing & awareness do not affect immediately in full
#define MAXDROPINELASTICNEED 0.98f //Even when marketing stops, needs drop only slowly as there is slowness in behavior patterns
#define MAXSTOCKLIMITINCREASE 1.2f //If stock limit increases quickly, it induces turbulence and bancrupties
#define MAXSTOCKLIMITDECREASE 0.95f //If stocklimit decreases sharply, temporary shortages may cause spoilage
#define MAXEFFICIENCYUPGRADE 1.05f // default 1.05 Learning is not immediate
#define MAXEFFICIENCYDOWNGRADE 0.98f //default 0.98 Learned capabilities remain and are forgotten slowly, also many other production resources remain
#define SWITCHTIME 1
//The following #define's are constants for documentation or readability
#define NETWORK 1 //Code, should be always on
//#define MARKET 2 //alternative trading mechanism not in use currently
#define NOMARKET 1
#define CONSUMER 10 //ID of CONSUMER, RETAILER & PRODUCER are used to identify the role that an individual has related to buying & selling each utility
#define RETAILER 11 //The ID also acts as a multiplier of exchange value index. Retailer should be higher than consumer and producer highest.
#define PRODUCER 12 //We rather exchange with retailer or producer all else being equal. Difference should be same order as transparency.
#define SURPLUSDEALS 1
#define CONSUMPTION 2
#define REGULARROUND 0
#define SURPLUSROUND 1
#define LEISUREROUND 2
#define TESTINDIVIDUAL (market.testindividual)
#define DEFAULTTESTINDIVIDUAL 10
#define TESTSKILL 1
#define TESTSKILL2 2
#define TESTSKILL3 3
#define TESTSKILL4 4
#define TESTSKILL5 5
#define TESTSKILL6 6
#define TESTSKILL7 7
#define TESTSKILL8 8
#define TESTSKILL9 9
#define TESTSKILL10 10



char mode = '1';

typedef struct { //The number of each dyadic transaction of each kind is documented and transparency calculated based on those and other parameters
	short relationid;
	short sold [SKILLS];
	short purchased [SKILLS];
	float transparency [SKILLS];
	float transactions;
}
EVENTS;

typedef struct { //Each agent contains private information on production capabilities, stock of utilities, friendship-network, value-expectations of utilities etc.
	EVENTS relationship [MAXGROUP];
	double efficiency [SKILLS]; //Calculated production efficiency
	int gifted [SKILLS]; // Random inborn talent - ON / OFF -type
	int role [SKILLS]; // Role can be consumer, producer or retailer, role changes are possible and role affects heuristics
	long long need [SKILLS]; //Need of each utility is reset each cycle and boosted if freetime
	long long surplus [SKILLS]; //Surplus stock of each utility, can be consumed or traded
	long long stocklimit[SKILLS]; //calculated maximum stock, recent incremented needs multiplied with maxsurplusfactor and recent sales added
	long long previousstocklimit[SKILLS];
	long long totalsurplus; //The total number of all kinds of items in stock, calculated after agent ends each cycle
	float salesprice [SKILLS]; //Minimum and maximum values required for given and received utilities for exchange to be profitable
	float purchaseprice [SKILLS];//Relation of utility given and received is meaningful, negotiation of exchange takes both agents values in account
	int purchasetimes [SKILLS];//Number of purchases per cycle is used for several purposes including following statistics
	float sumperiodpurchasevalue [SKILLS];
	int salestimes [SKILLS];
	float sumperiodsalesvalue [SKILLS];
	long long recentlyproduced [SKILLS];//Production statistics is used for calculating efficiency, transparency and reporting
	long long producedthisperiod[SKILLS];
	long long producedxperiod[SKILLS];
	long long recentlypurchased [SKILLS];
	long long purchasedthisperiod [SKILLS];
	long long purchasedxperiod [SKILLS];
	long long recentlysold [SKILLS];
	long long soldthisperiod [SKILLS];
	long long soldxperiod [SKILLS];
	long long spoils [SKILLS];//Excess surplus stock is partially spoiled each cycle and spoils are calculated each round per kind and summed
	long long periodicspoils;
	long long periodremaining;//periodremaining is reset each cycle
	long long periodremainingdebt;//last period overtime, deduced from next period
	int periodfailure; //if time runs out - similar to timeout but used within an agent (can possibly replace timeout but not vice versa)
	int timeout;//if time runs out and needs are not fulfilled timeout is flagged - this is supposed to be quite rare
//	float needsincrement;//if freetime, needs are increased temporarily and recent needsincrement is calculated
	float needslevel;
	float recentneedsincrement;
	int entrepreneur;//this is reserved for further use
}
PERSON;

PERSON individual [POPULATION];

typedef struct { //These are temporary variables for evaluating and storing best exchange possibilities
	float	index;
	int 	individualid;
	int 	friendid;
	int 	surplusproductid;
	float 	switchaverage;
	int 	exchangetype;
}
EXCHANGE;

EXCHANGE 		bestexchange;
float 			fexchangeindex[MAXGROUP][SKILLS];

typedef struct { //Used for evaluating friend candidate
	int index;
	int individualid;
	int friendid;
}
FRIENDCANDIDATE;

typedef struct { //Stored calculated constants. Variables for calculating statistics for reporting, increased needs and price elasticity
	long long period; //Number of cycles simulated so far
	long long periodlength;//Number of timeslots available during each period for each agent
	long long leisuretime;//constant for minimum unused timeslots required for calculating needsincrease, initialized
	long long basicneed[SKILLS];//storage for calculated constant, calculated value stored for efficiency reasons
	double elasticneed[SKILLS];//elastic need reflecting price elasticity but calculated based on production efficiency
	double previouselasticneed[SKILLS];//needed to limit elastic need change
	long long surplus [SKILLS];//calculated sum of all agent's surplus stock of each kind of utility at the end of each cycle
	double averageprice [SKILLS];//average production cost of each utility calculated each cycle
	double maxefficiency [SKILLS];//maximum production efficiency of each utility
	int purchasetimes [SKILLS];//The rest are used for statistics
	int salestimes [SKILLS];
	double sumperiodicpurchasevalue [SKILLS];
	double sumperiodsalesvalue [SKILLS];
	long long numberofrecentlyproduced[SKILLS];
	long long numberoftotalrecentproduction;
	long long producedthisperiod[SKILLS];
	long long periodictcecost[SKILLS];
	long long periodicspoils [SKILLS];
	long long costoftceintime[SKILLS];
	long long costofspoilsintime[SKILLS];
	long long totalcostoftceintime;
	long long totalcostofspoilsintime;
	double priceaverage;
//	long long totalsurplus;
	long long totalmisurplus;
	int numberofconsumers[SKILLS];
	int numberofretailers[SKILLS];
	int numberofproducers[SKILLS];
	int testindividual;
	int loosers;
}
MARKETSTRUCTURE;

MARKETSTRUCTURE market;

float InvSqrt (float x) //from http://www.geometrictools.com/Documentation/FastInverseSqrt.pdf
{
float xhalf = 0.5f*x;
int i = *(int*)&x;
i = 0x5f3759df - (i >> 1); // This line hides a LOT of math!
x = *(float*)&i;
x = x*(1.5f - xhalf*x*x); // repeat this statement for a better approximation
return x;
}

void initmarket (void){ //Market variables are initialized
	market.period = 1;
	market.periodlength = 0; //Periodlength is sum total of the number of utilities each agent needs and calculated in the loop
	for (int i=1;i<SKILLS;i++){
		market.basicneed [i] = (i * i * INTMULTIPLIER); //Needs are created and basic needs are similar to all agents
		market.elasticneed[i] = market.basicneed [i]; //Initializing elastic need should not be necessary but is done just in case
		market.previouselasticneed[i] = market.basicneed [i]; //Initializing elastic need should not be necessary but is done just in case
		market.periodlength = market.periodlength + market.basicneed[i];
//		market.surplus [i] = 0;
		market.averageprice [i] = 1.0f;
		market.purchasetimes [i] = 0;
		market.salestimes [i] = 0;
		market.sumperiodicpurchasevalue [i] = 0;
		market.sumperiodsalesvalue [i] = 0;
//		market.recentlypurchased [i] = 0;
//		market.recentlysold [i] = 0;
		market.numberofrecentlyproduced[i] = market.basicneed[i] * (HISTORY * POPULATION);//History is assumed to get realistic statistics
		market.producedthisperiod[i] = 0;
		market.periodictcecost[i] = 0;
		market.numberofconsumers[i]=0;
		market.numberofretailers[i]=0;
		market.numberofproducers[i]=0;
		market.testindividual = DEFAULTTESTINDIVIDUAL;
	}
	market.leisuretime = market.periodlength / SKILLS;
}

void initpopulation (void){ //Agents are initialized
	printf("initializing population");
	for (int i=1;i<POPULATION; i++)
	{
		individual[i].periodremaining = market.periodlength;
		individual[i].periodremainingdebt = 0;
		individual[i].needslevel = 1.0f;
//		individual[i].needsincrement = 0;
		individual[i].recentneedsincrement = 1.0f;
		individual[i].timeout = 0;

		for (int j=1; j<SKILLS; j++)
		{
			individual[i].need[j] = market.basicneed[j];
			individual[i].purchaseprice[j] = INITIALPRICE;
			individual[i].salesprice[j] = INITIALPRICE;
			individual[i].efficiency[j] = INITIALEFFICIENCY;
			individual[i].surplus[j] = market.basicneed[j]; //Surplus stock is set to what an agent can produce per cycle to avoid initial surge
			individual[i].stocklimit[j] = MAXSURPLUSFACTOR * market.basicneed[j];
			individual[i].previousstocklimit[j] = MAXSURPLUSFACTOR * market.basicneed[j];
			individual[i].gifted[j] = 0; //gifted are selected and initialized in a separate function for efficiency reasons
			individual[i].role[j] = CONSUMER;
			individual[i].recentlyproduced[j] = HISTORY * market.basicneed[j];//Production history is assumed as if each would produce their own need
			individual[i].producedthisperiod[j] = market.basicneed[j];
			individual[i].recentlysold[j] = 0; //Exchange history is initialized as if there were no previous exchange
			individual[i].soldthisperiod[j] = 0;
			individual[i].recentlypurchased[j] = 0;
			individual[i].purchasedthisperiod[j] = 0;
			individual[i].purchasetimes[j] = 0;
			individual[i].salestimes[j] = 0;
			individual[i].sumperiodpurchasevalue[j] = 1.0f;//Private values are set to initial ungifted production value
			individual[i].sumperiodsalesvalue[j] = 1.0f;
			individual[i].periodfailure = 0;
		}
		for (int k=1; k<MAXGROUP; k++) //Transaction data is initialized to no previous transactions
		{
			individual[i].relationship[k].transactions = 0;
			individual[i].relationship[k].relationid = 0;

			for (int l=1; l<SKILLS; l++)
			{
				individual[i].relationship[k].purchased[l]=0;
				individual[i].relationship[k].sold[l]=0;
				individual[i].relationship[k].transparency[l] = INITIALSOCIALTRANSPARENCY;
			}
		}
	}
}

int randomindividual(void){ //Random individual selected for giftedness or candidate for friend network
	int randindividual = rand();
	while (randindividual > POPULATION-1)
		randindividual = randindividual - POPULATION;
	return randindividual;
}

void selectgiftedpopulation (void){ //The number of gifts for each utility is at max maxgifted, and given at random
	for (int skill=1; skill<SKILLS; skill++)
	{
		for (int j=1; j<MAXGIFTED; j++)
		{
			int randindividual = randomindividual();

			individual[randindividual].efficiency[skill] = GIFTEDINITIALEFFICIENCY;
			individual[randindividual].gifted[skill] = TRUE;
		}
	}
}

int getmefriendindex(int me, int doiknowyou) //we need to convert the index from the proposer viewpoint to our own index
{
	int myfriend = MAXGROUP;
	while (--myfriend){
		if (individual[me].relationship[myfriend].relationid == doiknowyou)
			break;
	}
	return myfriend;
}

int getleastusefulfriendid (int me) //Least useful friend is selected unless free friendslots are available
{
	FRIENDCANDIDATE leastusefulfriend;
	int currentindex = MAXUSEFULINDEX; //maxuseful must be set to higher than lowest candidate always
	leastusefulfriend.index = currentindex;

	for (int myfriend = 1; myfriend < MAXGROUP; myfriend++)
	{
		if (individual[me].relationship[myfriend].relationid == FALSE)				{
				leastusefulfriend.friendid = myfriend;
			break;
		} else {
			currentindex = individual[me].relationship[myfriend].transactions; //only transactions are factored presently

			if (currentindex < leastusefulfriend.index)				{
				leastusefulfriend.index = currentindex;
				leastusefulfriend.friendid = myfriend;
			}
		}
	}
	return leastusefulfriend.friendid;
}

int makenewfriends(int me, int suggestedid){ //We come here from successful exchange proposal or our own wish to get a new friend

	//select the least useful friend for replacement or a free friendslot
	int leastusefulfriendid = getleastusefulfriendid(me);

	//let us select a random individual and hope it is a new aquintance
	int newfriendisnotsonew;
	int toomanyloops = 0;

	//if we have not been approached by our friend, we select randomly a new aquintance and check that it really is "new"
	if (suggestedid == FALSE) {
		do {
			newfriendisnotsonew = FALSE;
			suggestedid = randomindividual();
			if (suggestedid == me) newfriendisnotsonew = TRUE;
				else {
							for (int somefriend = 1; somefriend < MAXGROUP; somefriend++)
								{
								if (individual[me].relationship[somefriend].relationid == suggestedid)					{
									newfriendisnotsonew = TRUE;
								break;
								}
							}
			}
			if (toomanyloops++ > 2 * MAXGROUP) { //this is to rule out bugs and contradictory parameters
				printf("toomanyloops, suggestedid %i ", suggestedid);
			}
		}
		while(newfriendisnotsonew);
	}

	int newfriendid = leastusefulfriendid; //New friend is placed into the slot of the least useful friend and transaction data is reset
	individual[me].relationship[newfriendid].relationid = suggestedid;
	individual[me].relationship[newfriendid].transactions = 2;

	if (me == TESTINDIVIDUAL) printf("me %i, mynewfriend %i, friendid %i \n", me, suggestedid, newfriendid);
	if (suggestedid == TESTINDIVIDUAL) printf("indy%i, I am his new friend %i, friendid %i \n", me, suggestedid, newfriendid);

	for (int l=1; l<SKILLS; l++)
	{
		individual[me].relationship[newfriendid].purchased[l]=0;
		individual[me].relationship[newfriendid].sold[l]=0;
		individual[me].relationship[newfriendid].transparency[l] = INITIALSOCIALTRANSPARENCY;
	}
	return newfriendid;
}

void checksurplus (int debug){ //Testing that surplus is positive
//	for (int i=1;i<POPULATION; i++)
//	{
//		for (int j=1; j<SKILLS; j++)
//		{
////			if (individual[i].surplus[j] < 0) {
//			if (individual[i].surplus[j] > 10*individual[i].stocklimit[j]) {
//
//				printf("checksurplus - surplus negative %lli, setting to zero, debug code %i utility %i individual %i", individual[i].surplus[j], debug, j, i);
//				individual[i].surplus[j] = 0;
//			}
//		}
//	}
}


float calculatemarketprice (int need)
//marketprice of a given utility is calculated as the weighed production cost of recent production and affects price elasticity.
{
	float marketprice = 0.0f;
	double sumofproductioncost = 0.0f;

	market.producedthisperiod[need] = 0;
	market.maxefficiency[need] = 0.0f;

	for (int indy = 1; indy < POPULATION; indy++){
		individual[indy].totalsurplus = individual[indy].totalsurplus + individual[indy].surplus[need];

		sumofproductioncost = sumofproductioncost + ((double)individual[indy].producedthisperiod[need] / individual[indy].efficiency[need]);
		market.producedthisperiod[need] = market.producedthisperiod[need] + individual[indy].producedthisperiod[need];
		if (individual[indy].efficiency[need] > market.maxefficiency[need])
			market.maxefficiency[need] = individual[indy].efficiency[need];
		if (individual[indy].role[need] == CONSUMER) market.numberofconsumers[need]++;
		if (individual[indy].role[need] == RETAILER) market.numberofretailers[need]++;
		if (individual[indy].role[need] == PRODUCER) market.numberofproducers[need]++;

	}
	//market price is not affected if production is
	if (market.producedthisperiod[need] > (POPULATION * market.elasticneed[need])){
		marketprice = (float)((((double)market.producedthisperiod[need] /
				(double)(market.producedthisperiod[need] + market.numberofrecentlyproduced[need])) *
					(sumofproductioncost / (double)market.producedthisperiod[need])) +
					(((double)market.numberofrecentlyproduced[need] /
					(double)(market.producedthisperiod[need] + market.numberofrecentlyproduced[need])) *
						(double)market.averageprice[need]));
				} else marketprice = market.averageprice[need];


	market.numberofrecentlyproduced[need] = market.numberofrecentlyproduced[need] + market.producedthisperiod[need];
	if (marketprice < 0 || marketprice > 100) printf("marketprice negative or over 100, marketprice %f!!!!!!!!!!!!", marketprice);

	return marketprice;
}

void evaluatemarketprices (void) //average production cost of all utilities is calculated and price elasticity of utilities
{

	market.priceaverage = 0;
	market.numberoftotalrecentproduction = 0;
	market.totalcostofspoilsintime = 0;
	market.totalcostoftceintime = 0;
	for (int i=1; i<SKILLS;i++){
		market.numberofconsumers[i]=0;
		market.numberofretailers[i]=0;
		market.numberofproducers[i]=0;

		market.averageprice[i] = calculatemarketprice(i);
		market.numberoftotalrecentproduction = market.numberoftotalrecentproduction + market.numberofrecentlyproduced[i];
		market.priceaverage = market.priceaverage + (market.averageprice[i] * market.numberofrecentlyproduced[i]);

	}

	market.priceaverage = market.priceaverage / market.numberoftotalrecentproduction;
	long long totalelastic =0;
	for (int i=1; i<SKILLS;i++){ //Utility statistics is calculated
		market.previouselasticneed[i] = market.elasticneed[i];

		switch (PRICEDEMANDELASTICITY){
		case 0:	market.elasticneed[i] = market.basicneed[i];
			break;
		case 1:market.elasticneed[i] = (market.basicneed[i] * ((market.priceaverage / market.averageprice[i])));
			break;
		case 2:market.elasticneed[i] = (market.basicneed[i] * ((market.priceaverage / market.averageprice[i])*(market.priceaverage / market.averageprice[i])));
			break;
		default: printf("mistaken value in PRICEDEMANDELASTICITY");
		}
		totalelastic = totalelastic+market.elasticneed[i];
		market.costofspoilsintime[i] = (market.periodicspoils[i] * market.averageprice[i]);
		market.totalcostofspoilsintime = market.totalcostofspoilsintime + market.costofspoilsintime[i];
		market.costoftceintime[i] = (market.periodictcecost[i] * market.averageprice[i]);
		market.totalcostoftceintime = market.totalcostoftceintime + market.costoftceintime[i];
	}
	long long totalelastic2 = 0;
	for (int i=1; i<SKILLS;i++){ //Elastic need is calibrated, sum equal to periods, and then utility statistics per cycle data is printed
		market.elasticneed[i] = market.elasticneed[i] * ((double)market.periodlength / totalelastic);

		//Limit maxrise and maxdrop of elasticneed, this may affect total elastic need but should not affect it much
		//This effect can be removed by repeating previous for-loop if more accuracy is required .... and well - it was required :-(
		if (market.elasticneed[i]>MAXRISEINELASTICNEED * market.previouselasticneed[i])
			market.elasticneed[i]=MAXRISEINELASTICNEED * market.previouselasticneed[i];
		if (market.elasticneed[i]<MAXDROPINELASTICNEED * market.previouselasticneed[i])
			market.elasticneed[i]=MAXDROPINELASTICNEED * market.previouselasticneed[i];
		totalelastic2=totalelastic2+market.elasticneed[i];
	}
	long long totalelastic3 = 0;
	for (int i=1; i<SKILLS;i++){ //Elastic need is calibrated, sum equal to periods, and then utility statistics per cycle data is printed
		market.elasticneed[i] = market.elasticneed[i] * ((double)market.periodlength / totalelastic2);
		totalelastic3=totalelastic3+market.elasticneed[i];


		printf("need %i prodcost %.4f elasticneed %.1f maxeff %.2f prodnow/pEneed %.2f recentprod/prEneed %.2f COspoilsPEneed %.2f COtcecostPEneed %.2f, Cons %.3f Ret %.3f Prod %.3f\n",
			i, market.averageprice[i], (float)market.elasticneed[i], market.maxefficiency[i],
			(float)market.producedthisperiod[i]/(POPULATION*market.elasticneed[i]),(float)market.numberofrecentlyproduced[i]/(POPULATION*market.elasticneed[i]*(HISTORY+1)),
		    (float)(((market.averageprice[i] / market.priceaverage)*market.periodicspoils[i]) / (market.elasticneed[i]*POPULATION)),
			(float)(((market.averageprice[i] / market.priceaverage)*market.periodictcecost[i]) / (market.elasticneed[i]*POPULATION)),
			(float)market.numberofconsumers[i]/POPULATION,(float)market.numberofretailers[i]/POPULATION,(float)market.numberofproducers[i]/POPULATION);
	}
	printf("totalelastic %lli %lli %lli difference of 3. to periodlength %lli\n", totalelastic, totalelastic2, totalelastic3, totalelastic3-market.periodlength);
}

void getfriendexchangeindexes(int me, int myneed) //All exchange options for myneed from all friends are indexed and best option selected
{
	int myfriend;
	int measfriendindex;
	int positiveexchanges; //used to count the number of non CONSUMER -positive ... , but not implemented yet?
	float receivingtransparency;
	for (int friend=1; friend < MAXGROUP; friend++) //checking all existing social connections
	{
		myfriend = individual[me].relationship[friend].relationid;
		measfriendindex = getmefriendindex(myfriend, me);

		if (myfriend > 0)	{
				for (int mygift=1; mygift < SKILLS; mygift++) //checking all surplused exchange options
				// individual "me" wants "myneed" and offers "mygift", relationship "myfriend" accepts "mygift"
				// and yields "myneed". Transparency decreases potential yield for "me" as a percentage of loss
				// and is calculated here as a corresponding increase of purchase price
				{
					// we only need to evaluate exchanges with adequate surpluses
//							printf("Kilroy %i k‰vi t‰‰ll‰", me);
					int giftmaxlevel = 0;
					if (individual[myfriend].role[mygift] == RETAILER)
						giftmaxlevel = individual[myfriend].stocklimit[mygift] - INTMULTIPLIER;
					else giftmaxlevel = (market.elasticneed[mygift]*individual[myfriend].needslevel)-INTMULTIPLIER;

					if ((individual[myfriend].surplus[myneed] > ((market.elasticneed[myneed]*individual[myfriend].needslevel) + (INTMULTIPLIER))) &&
		//			    (individual[myfriend].surplus[mygift] < (individual[myfriend].stocklimit[mygift] - market.basicneed[1]))
		//			    (individual[myfriend].surplus[mygift] < ((market.elasticneed[mygift]*individual[myfriend].needslevel)-(INTMULTIPLIER)))
						(individual[myfriend].surplus[mygift] < giftmaxlevel)
						&& (individual[me].surplus[mygift] > (market.elasticneed[mygift]*individual[me].needslevel)+INTMULTIPLIER)){
							//printf("Kilroy %i k‰vi t‰‰ll‰ kans", me);
							if (measfriendindex == 0) receivingtransparency = INITIALSOCIALTRANSPARENCY;
								else receivingtransparency = individual[myfriend].relationship[measfriendindex].transparency[mygift];

							//fexchangeindex is positive if both parties jointly gain in exchange i.e. if positive sum game results
							fexchangeindex[friend][mygift] = ((individual[myfriend].purchaseprice[mygift] /
							individual[myfriend].salesprice[myneed]) *
						    individual[me].relationship[friend].transparency[myneed]) -
						    (individual[me].salesprice[mygift] / (individual[me].purchaseprice[myneed] * receivingtransparency));
							//We favour those who are producers and retailers by using role-value as a multiplier !!!!!!!!!!!!!!!!!!!!!!!!!
							//This is a question of search priority and about friend not having an opportunity to select best option
							//but this implementation method is not perfect, however we do have these tendencies to favour those
							//who have clear and established motives and routines to sell - and this is separate from transparency
							//Multiplier values are defined as CONSUMER, RETAILER & PRODUCER and if stepping is too sparse, it
							//overrides the effect of full transparency, which clearly is not realistic.
							fexchangeindex[friend][mygift]=individual[me].role[mygift] * fexchangeindex[friend][mygift];
							fexchangeindex[friend][mygift]=individual[myfriend].role[myneed] * fexchangeindex[friend][mygift];
							if (individual[me].role[myneed]==PRODUCER) fexchangeindex[friend][mygift]= fexchangeindex[friend][mygift]/2;
						//debug
//							if (me == TESTINDIVIDUAL)								{
//								printf("exchindex %f for need %i from friend %i\n",fexchangeindex[friend][mygift], myneed, myfriend);
//								printf("myfreceivespp %f myfgivessp %f transp %f mygiftsp %f mereceivepp %f \n", individual[myfriend].purchaseprice[mygift],
//									individual[myfriend].salesprice[myneed],
//									individual[me].relationship[friend].transparency[myneed],
//									individual[me].salesprice[mygift], individual[me].purchaseprice[myneed]);
//							}
						}
						else fexchangeindex[friend][mygift] = -1;

					if (bestexchange.index < fexchangeindex[friend][mygift])							{
						bestexchange.index = fexchangeindex[friend][mygift];
						bestexchange.individualid = myfriend;
						bestexchange.friendid = friend;
						bestexchange.surplusproductid = mygift;
						bestexchange.exchangetype =NETWORK;
					}
				}
		} else
				for (int mygift = 1;mygift < SKILLS; mygift++)
					fexchangeindex[friend][mygift] = -1.0f;
	}
}


void excecutefexchange (int dealtype, int me, int myneed, long long maxneed)
//best selected exchange option is calculated in friendexchangeindexes and stored partially in bestexchange and partially in params
//this function calculates exchange relation and defines maximum exchange and commits the exchange
{
	long long maxexchange;
	int mygift = bestexchange.surplusproductid;
	int measfriendindex = getmefriendindex(bestexchange.individualid, me);
	float receivingtransparency;

	if (measfriendindex == 0) receivingtransparency = INITIALSOCIALTRANSPARENCY;
		else receivingtransparency = individual[bestexchange.individualid].relationship[measfriendindex].transparency[mygift];


//		bestexchange.switchaverage = (((individual[bestexchange.individualid].purchaseprice[mygift] /	//careful again, partially tested
//			individual[bestexchange.individualid].salesprice[myneed]) *									//only approximation, not pareto optimal
//		    individual[me].relationship[bestexchange.friendid].transparency[myneed]) +
//		    (individual[me].purchaseprice[myneed] / individual[me].salesprice[mygift]) * receivingtransparency)/2;
		bestexchange.switchaverage = (((individual[bestexchange.individualid].salesprice[myneed] /	//
			(individual[bestexchange.individualid].purchaseprice[mygift]*individual[me].relationship[bestexchange.friendid].transparency[myneed])) +
		    ((individual[me].purchaseprice[myneed]*receivingtransparency) / individual[me].salesprice[mygift])))/2; //!!!!!!!Transparency missing!!!!!!!!!!!!!!!!!

	//now commit exchange and update sales&purchaseprices
	//Following several lines limit maxexchange - starting with mygift surplus
	maxexchange = (long long) (((individual[me].surplus[mygift] - (individual[me].needslevel * market.elasticneed[mygift])) * receivingtransparency)/ bestexchange.switchaverage);
	if (maxexchange <= (INTMULTIPLIER/2)) //discard too small lot to be of any use in exchange!!!!!
		{
	//	individual[me].surplus[mygift] = 0;
		maxexchange = 0;
		for (int i = 1; i < MAXGROUP; i++){
			fexchangeindex[i][bestexchange.surplusproductid] = -1.0f;
		} //saving a waste of time for the rest of friends canceling a promise for a nonsurplussed gift
		} else {
			if (maxexchange > maxneed)
				maxexchange = maxneed;
			if (((individual[bestexchange.individualid].surplus[myneed] - (individual[bestexchange.individualid].needslevel * market.elasticneed[myneed]))* //enough surplus in the sales side
				individual[me].relationship[bestexchange.friendid].transparency[myneed]) < maxexchange)

				maxexchange = 	(long long) ((individual[bestexchange.individualid].surplus[myneed]-
								(individual[bestexchange.individualid].needslevel * market.elasticneed[myneed])) *
				    			individual[me].relationship[bestexchange.friendid].transparency[myneed]);

			if (individual[bestexchange.individualid].role[mygift] == RETAILER){
				if ((individual[bestexchange.individualid].stocklimit[mygift] -
					individual[bestexchange.individualid].surplus[mygift]) < (long long) maxexchange * bestexchange.switchaverage)
					maxexchange = 	(long long)((individual[bestexchange.individualid].stocklimit[mygift] -
									 individual[bestexchange.individualid].surplus[mygift])/bestexchange.switchaverage);
			} else //if not retailer, we only supply for immediate need
				if (((individual[bestexchange.individualid].needslevel * market.elasticneed[mygift]) -
					individual[bestexchange.individualid].surplus[mygift]) < (long long) maxexchange * bestexchange.switchaverage)
					maxexchange = 	(long long)(((individual[bestexchange.individualid].needslevel * market.elasticneed[mygift]) -
									 individual[bestexchange.individualid].surplus[mygift])/bestexchange.switchaverage);
//!!!!!maxexchange evaluated, BUT some calculation brings maxexchange too close to causing negative surplus !!!
			maxexchange=maxexchange - INTMULTIPLIER/3; //getting rid of rounding errors and potential negative values for surplus due to too tight calculations
//			if (me==TESTINDIVIDUAL) printf("maxexchange is %lli friend is %i \n", maxexchange, bestexchange.individualid);
			if (measfriendindex == 0){
				measfriendindex = makenewfriends(bestexchange.individualid, me);
			}
		if (maxexchange >= (INTMULTIPLIER/2)) {
			maxneed = maxneed - maxexchange;
			if (dealtype == SURPLUSDEALS) { //if we are optimizing our surplus, not satisfying needs
				individual[me].surplus[myneed] = individual[me].surplus[myneed] + maxexchange;
				if (individual[me].surplus[myneed] > individual[me].stocklimit[myneed]) printf("surplusdeals ... stocklimit exceeded for myneed!!!");
			} else	{
				individual[me].need[myneed] = (individual[me].need[myneed] - maxexchange);
				if (individual[me].need[myneed] < INTMULTIPLIER/2) individual[me].need[myneed]=0;
				}
			if (individual[me].stocklimit[myneed] < individual[me].surplus[myneed]) printf("surplus above stocklimit due to maxexchange overflow??");
			individual[me].recentlysold[mygift] = (individual[me].recentlysold[mygift] + (long long)(maxexchange * bestexchange.switchaverage));
			individual[me].soldthisperiod[mygift] = (individual[me].soldthisperiod[mygift] + (long long)(maxexchange * bestexchange.switchaverage));
			individual[me].recentlypurchased[myneed] = (individual[me].recentlypurchased[myneed] + maxexchange);
			individual[me].purchasedthisperiod[myneed] = (individual[me].purchasedthisperiod[myneed] + maxexchange);
			individual[me].surplus[mygift] = (individual[me].surplus[mygift] - (long long)((double)(maxexchange * bestexchange.switchaverage) / receivingtransparency));
			if (individual[me].surplus[mygift]<0) printf("mygift surplus negative when doing exchange");
			if ((maxneed >= INTMULTIPLIER) && (individual[me].surplus[mygift] < INTMULTIPLIER)){
				 //discard too small lot to be of any use in exchange!!!!!
				individual[me].surplus[mygift] = 0;
				for (int i = 1; i < MAXGROUP; i++){
					fexchangeindex[i][mygift] = -1.0f;
				} //saving a waste of time for the rest of friends canceling a promise for a nonsurplussed gift
			}

			individual[bestexchange.individualid].surplus[myneed] =
			    (individual[bestexchange.individualid].surplus[myneed] - (long long)(maxexchange /
				individual[me].relationship[bestexchange.friendid].transparency[myneed]));

			market.periodictcecost[myneed] = market.periodictcecost[myneed] +
											((maxexchange / individual[me].relationship[bestexchange.friendid].transparency[myneed]) - maxexchange);

			market.periodictcecost[mygift] = market.periodictcecost[mygift] +
											(((double)(maxexchange * bestexchange.switchaverage) / receivingtransparency) - maxexchange * bestexchange.switchaverage);

			individual[bestexchange.individualid].recentlysold[myneed] =
			    (individual[bestexchange.individualid].recentlysold[myneed] + (long long)(maxexchange /
				individual[me].relationship[bestexchange.friendid].transparency[myneed]));
			individual[bestexchange.individualid].soldthisperiod[myneed] =
			    (individual[bestexchange.individualid].soldthisperiod[myneed] + (long long)(maxexchange /
				individual[me].relationship[bestexchange.friendid].transparency[myneed]));

			individual[bestexchange.individualid].recentlypurchased[mygift] =
			    (individual[bestexchange.individualid].recentlypurchased[mygift] + (long long)(maxexchange * bestexchange.switchaverage));
			individual[bestexchange.individualid].purchasedthisperiod[mygift] =
			    (individual[bestexchange.individualid].purchasedthisperiod[mygift] + (long long)(maxexchange * bestexchange.switchaverage));

			individual[bestexchange.individualid].surplus[mygift] =
			    (individual[bestexchange.individualid].surplus[mygift] + (long long)(maxexchange * bestexchange.switchaverage));
			if (individual[bestexchange.individualid].surplus[mygift] > (individual[bestexchange.individualid].stocklimit[mygift]*MAXNEEDSINCREASE))
				printf("surplusdeals ... myfriends stocklimit %lli exceeded, surplus %lli while need is %lli for mygift!!!\n",
				individual[bestexchange.individualid].stocklimit[mygift],individual[bestexchange.individualid].surplus[mygift],
				(long long) (market.elasticneed[mygift] * individual[bestexchange.individualid].needslevel));

			float exchangevaluetome =(((float)individual[me].purchaseprice[myneed])/((bestexchange.switchaverage / receivingtransparency)*individual[me].salesprice[mygift]));
			float exchangevaluetofriend =(((float)individual[bestexchange.individualid].purchaseprice[mygift]*bestexchange.switchaverage) /
				(individual[bestexchange.individualid].salesprice[myneed]/individual[me].relationship[bestexchange.friendid].transparency[myneed]));
			if (me == TESTINDIVIDUAL) printf("surplusdeals myneed %i bestexchangeindex %f, myxvalue (average >1) %f fxchvalue (average >1) %f\n",
				myneed, bestexchange.index, exchangevaluetome,exchangevaluetofriend);

			//now the perceived utility values should be adjusted - salesprices raised and purchaseprices lowered by equal proportion
			//so that for both parties new sales & purchaseprices would yield exchangeindex == 0 for this committed exchange
			//thus preventing any exchanges that are not as useful. This is however not done based on one single deal
			//instead statistic is collected of exchange value for present price-levels and price is adjusted based on
			//the performance of the whole period and remaining surplus - current calculation is not weighed per deal size
			individual[me].purchasetimes[myneed]++;
			individual[me].salestimes[mygift]++;
			individual[bestexchange.individualid].purchasetimes[mygift]++;
			individual[bestexchange.individualid].salestimes[myneed]++;
			float myvaluecorrection = 1.0f / InvSqrt(exchangevaluetome);
			float friendvaluecorrection = 1.0f / InvSqrt(exchangevaluetofriend);


			if ((myvaluecorrection > 1)	&& (friendvaluecorrection > 1))					{
				individual[me].sumperiodpurchasevalue[myneed]= (individual[me].sumperiodpurchasevalue[myneed] +
				    individual[me].purchaseprice[myneed] / myvaluecorrection);
				individual[me].sumperiodsalesvalue[mygift] = (individual[me].sumperiodsalesvalue[mygift] +
				    individual[me].salesprice[mygift] * myvaluecorrection);
				individual[bestexchange.individualid].sumperiodpurchasevalue[mygift] = (individual[bestexchange.individualid].sumperiodpurchasevalue[mygift] +
				    individual[bestexchange.individualid].purchaseprice[mygift] / friendvaluecorrection);
				individual[bestexchange.individualid].sumperiodsalesvalue[myneed] = (individual[bestexchange.individualid].sumperiodsalesvalue[myneed] +
				    individual[bestexchange.individualid].salesprice[myneed] * friendvaluecorrection);
//				individual[me].sumperiodpurchasevalue[myneed]= (individual[me].sumperiodpurchasevalue[myneed] +
//				    individual[me].purchaseprice[myneed] / correction);
//				individual[me].sumperiodsalesvalue[mygift] = (individual[me].sumperiodsalesvalue[mygift] +
//				    individual[me].salesprice[mygift] * correction);
//				individual[bestexchange.individualid].sumperiodpurchasevalue[mygift] = (individual[bestexchange.individualid].sumperiodpurchasevalue[mygift] +
//				    individual[bestexchange.individualid].purchaseprice[mygift] / correction);
//				individual[bestexchange.individualid].sumperiodsalesvalue[myneed] = (individual[bestexchange.individualid].sumperiodsalesvalue[myneed] +
//				    individual[bestexchange.individualid].salesprice[myneed] * correction);
				} else { printf("mistaken valuecorrection - !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
//					individual[me].sumperiodpurchasevalue[myneed] = (individual[me].sumperiodpurchasevalue[myneed] +
//					    individual[me].purchaseprice[myneed] * correction);
//					individual[me].sumperiodsalesvalue[mygift] = (individual[me].sumperiodsalesvalue[mygift] +
//					    individual[me].salesprice[mygift] / correction);
//					individual[bestexchange.individualid].sumperiodpurchasevalue[mygift] = (individual[bestexchange.individualid].sumperiodpurchasevalue[mygift] +
//					    individual[bestexchange.individualid].purchaseprice[mygift] * correction);
//					individual[bestexchange.individualid].sumperiodsalesvalue[myneed] = (individual[bestexchange.individualid].sumperiodsalesvalue[myneed] +
//					    individual[bestexchange.individualid].salesprice[myneed] / correction);
			}

			//update transactions statistics to each need specific dyadic relation
			individual[me].relationship[bestexchange.friendid].transactions++;
			individual[me].relationship[bestexchange.friendid].purchased[myneed]++;
			;
			individual[me].relationship[bestexchange.friendid].sold[mygift]++;
			individual[bestexchange.individualid].relationship[measfriendindex].transactions++;
			individual[bestexchange.individualid].relationship[measfriendindex].purchased[mygift]++;
			;
			individual[bestexchange.individualid].relationship[measfriendindex].sold[myneed]++;
		}
	}
	//delete used fexchangeindex
	fexchangeindex[bestexchange.friendid][bestexchange.surplusproductid] = -1.0f;

}


int satisfyneedsbyexchange(int me){
	//we satisfy unsatisfied needs by exchange if affordable surplus available

	int surpluscoversallneeds = 1;
	for (int myneed=1; myneed<SKILLS; myneed++)
	{
		bestexchange.index = -1;
		bestexchange.surplusproductid = 0;

//		if (individual[me].need[myneed] >= (market.basicneed[1]/2)) //only unsatisfied needs processed here
		if (individual[me].surplus[myneed] < (market.elasticneed[myneed] * individual[me].needslevel)) //only unsatisfied needs processed here
			{

				//Next we create exchangeindexes from myneed to mygift for all friends and market and select bestexchange
				getfriendexchangeindexes(me, myneed);
				if ((individual[me].surplus[bestexchange.surplusproductid] < INTMULTIPLIER) && (bestexchange.index > 0))
					printf ("getfriendexcind brings back lacking surplus for mygift, bestexchangeindex %f surplusprid %i\n", bestexchange.index, bestexchange.surplusproductid);

				int toomanyloops = 0;
//				printf("t‰ss‰ oltiin myneed %i, bestexchangeindex %f \n", myneed, bestexchange.index);

				while ((bestexchange.index > 0) && (individual[me].need[myneed] >= INTMULTIPLIER))
//				while ((bestexchange.index > 0) && (individual[me].surplus[myneed] < market.basicneed[1]+(market.elasticneed[myneed] * individual[me].needslevel)))
				{
					if (toomanyloops++ > (SKILLS * MAXGROUP)) {
						printf ("its me %i and exchange is running hot, bestexchindex is %f\n",me, bestexchange.index);
						break;
					}
					if (individual[me].surplus[bestexchange.surplusproductid] < (INTMULTIPLIER))
						printf ("1 while loop brings back lacking surplus for me %i and mygift, bestexchangeindex %f surplusprid %i, surplus %lli, basicneed %lli\n",
								me, bestexchange.index, bestexchange.surplusproductid, individual[me].surplus[bestexchange.surplusproductid], market.basicneed[1]);

					if (bestexchange.exchangetype == NETWORK)
							excecutefexchange(CONSUMPTION, me, myneed, individual[me].need[myneed]);
//							excecutefexchange(SURPLUSDEALS, me, myneed, individual[me].need[myneed]);
//							excecutefexchange(SURPLUSDEALS, me, myneed, (individual[me].needslevel * market.elasticneed[myneed] - individual[me].surplus[myneed]));
							else {printf("nonexistent optional market mechanism selected in satisfyneedsbyexchange");
							//excecutemexchange(CONSUMPTION, me, myneed, individual[me].need[myneed]);
					}

					//initialize search for the next best exchange
					bestexchange.index = - 0.5f;
					bestexchange.surplusproductid = 0;

					// if this need is not satisfied, we need to try and do more exchange
					for (int friend=1; friend<MAXGROUP; friend++) {  //checking all existing social connections
						for (int gift=1; gift<SKILLS; gift++) //checking all surplused exchange options
							{
								if ((bestexchange.index < fexchangeindex[friend][gift]) && (individual[me].surplus[gift] > INTMULTIPLIER))
									{//checking surplus here should not be necessary but some nonsurplussed transactions arise possibly due to some bug
										bestexchange.index = fexchangeindex[friend][gift];
										bestexchange.individualid = individual[me].relationship[friend].relationid;
										bestexchange.friendid = friend;
										bestexchange.surplusproductid = gift;
										bestexchange.exchangetype =NETWORK;
								}
						}
					}


				}//endwhile - all potential exchanges for myneed exhausted
		} //if ended here

	}// all needs satisfied if possible by exchange
	for (int myneed=1; myneed<SKILLS; myneed++){
		if (individual[me].surplus[myneed]<(individual[me].needslevel * market.elasticneed[myneed])){
			surpluscoversallneeds = 0;
			break;
		}
	}
	return surpluscoversallneeds;
}

void makesurplusdeals (int me)
{
	//we select products where our recent sales exceeds our recent production and surplus is not at maximum
	//then we try to find deals that yield positive exchange ratio but we do not iterate on a single round

	for (int myneed=1; myneed<SKILLS; myneed++)
	{
		bestexchange.index = -1;
		bestexchange.surplusproductid = 0;

		if ((individual[me].recentlysold[myneed] > (individual[me].recentlyproduced[myneed] - market.elasticneed[myneed])) &&
			(individual[me].surplus[myneed] < (individual[me].stocklimit[myneed]) - market.elasticneed[myneed])) //only unsatisfied needs processed here
			{
					getfriendexchangeindexes(me, myneed);
					if ((individual[me].surplus[bestexchange.surplusproductid] < (INTMULTIPLIER)) && (bestexchange.index >0))
						printf ("getfriendexcind in spldls brings back lacking surplus for mygift, bestexchangeindex %f surplusprid %i\n", bestexchange.index, bestexchange.surplusproductid);
					int toomanyloops = 0;
//!!!Check if the limit in while should be basicneed smaller
					while ((bestexchange.index > 0) && (individual[me].surplus[myneed] < (individual[me].stocklimit[myneed])))
					{
						if (toomanyloops++ > (SKILLS * MAXGROUP)) {
							printf ("its me %i and exchange is running hot in mksplsdls, bestexchindex is %f\n",me, bestexchange.index);
							break;
						}
						if (individual[me].surplus[bestexchange.surplusproductid] < INTMULTIPLIER)
							printf ("2while loop brings back lacking surplus for mygift, bestexchangeindex %f surplusprid %i, exchtype %i\n",
									bestexchange.index, bestexchange.surplusproductid, bestexchange.exchangetype);

						if (bestexchange.exchangetype == NETWORK){
								excecutefexchange(SURPLUSDEALS, me, myneed, (long long)(individual[me].stocklimit[myneed] - individual[me].surplus[myneed]));
//								if (me == TESTINDIVIDUAL) printf("surplusdeals myneed %i bestexchangeindex %f, myexchangevalue %f myfriendexchvalue %f\n", myneed, bestexchange.index);
							}
							else {printf("nonexistent market mechanism selected in makesurplusdeals");
								//excecutemexchange(SURPLUSDEALS, me, myneed, ((MAXSURPLUSFACTOR * market.elasticneed[myneed]) + individual[me].recentlysold[myneed]) - individual[me].surplus[myneed]);
						}

						//initialize search for the next best exchange
						bestexchange.index = - 0.5f;
						bestexchange.surplusproductid = 0;

						// if this need is not satisfied, we need to try and do more exchange
						for (int friend=1; friend<MAXGROUP; friend++) {  //checking all existing social connections
							for (int gift=1; gift<SKILLS; gift++) //checking all surplused exchange options
								{
									if ((bestexchange.index < fexchangeindex[friend][gift]) && (individual[me].surplus[gift] > INTMULTIPLIER))
										{//checking surplus here should not be necessary but some nonsurplussed transactions arise possibly due to some bug
											bestexchange.index = fexchangeindex[friend][gift];
											bestexchange.individualid = individual[me].relationship[friend].relationid;
											bestexchange.friendid = friend;
											bestexchange.surplusproductid = gift;
											bestexchange.exchangetype =NETWORK;
									}
							}
						}


					}//endwhile - all potential exchanges for myneed exhausted
				//} // if ended here
		} //if ended here

	}// all needs satisfied if possible by exchange
}

void produceneed(int me){
	for (int myneed = 1; myneed < SKILLS; myneed++)
	{

		if (individual[me].need[myneed] > 0)				{
				individual[me].periodremaining = (individual[me].periodremaining - ((float)individual[me].need[myneed] /
					individual[me].efficiency[myneed]));
				individual[me].recentlyproduced[myneed] = individual[me].recentlyproduced[myneed] + individual[me].need[myneed];
				individual[me].producedthisperiod[myneed] = individual[me].producedthisperiod[myneed] + individual[me].need[myneed];
				if(me == TESTINDIVIDUAL) printf("Producing myneed%i %lli elasticneed %lli periodremaining %lli producednow %lli\n", myneed, individual[me].need[myneed], (long long)market.elasticneed[myneed], individual[me].periodremaining, individual[me].producedthisperiod[myneed]);
				individual[me].need[myneed]=0;
			}
	}
	if (individual[me].periodremaining < 0) {
			//printf("error, needs not fully satisfied and remaining time ended!");
			individual[me].timeout++; //needs reduced as consumption was in unsustainable level
		}
}

void producesmallneeds(int me){
	for (int myneed = 1; myneed < SKILLS; myneed++)
	{

		if ((individual[me].need[myneed] > 0) && (individual[me].need[myneed] < INTMULTIPLIER))				{
				individual[me].periodremaining = (individual[me].periodremaining - ((float)individual[me].need[myneed] /
					individual[me].efficiency[myneed]));
				individual[me].recentlyproduced[myneed] = individual[me].recentlyproduced[myneed] + individual[me].need[myneed];
				individual[me].producedthisperiod[myneed] = individual[me].producedthisperiod[myneed] + individual[me].need[myneed];
				individual[me].need[myneed]=0;
				if(me == TESTINDIVIDUAL) printf("Producing mysmallneed%i periodrem %lli producednow %lli\n", myneed, individual[me].periodremaining, individual[me].producedthisperiod[myneed]);
			}
	}
}


void surplusproduction(int me, int needtype){
	int selectedproduction;
	long long productionlimit;
	long long productionlimitselectedproduction=0;
	long long maxproduction;
	float productionindex;
	float currentproductionindex; //not yet in use - production index may be multiplied by efficiency to favour investment in experience
	long long maxtimeforsurplus;

	while (individual[me].periodremaining >= 1)		{
			productionlimit =0;
			productionindex = 0;
			selectedproduction = 0;
			maxtimeforsurplus = 0;

			for (int myneed=1; myneed<SKILLS; myneed++)
			{
				if (individual[me].gifted[myneed] == TRUE)	{ //for needs where you have no skills, surplus is not produced & basic round, surplus only to the extent it was necessary previous round
					productionlimit = ((individual[me].soldxperiod[myneed]+individual[me].soldthisperiod[myneed])+
						(((MAXNEEDSINCREASE*individual[me].needslevel)*market.elasticneed[myneed])-individual[me].surplus[myneed]));
//					productionlimit = individual[me].stocklimit[myneed]; //Testing if this has an effect
//					if ((needtype == SURPLUSROUND)&&(individual[me].recentlysold[myneed] > individual[me].stocklimit[myneed]/2)) //Testing the effect of
					if (needtype == SURPLUSROUND)
					    productionlimit = individual[me].stocklimit[myneed]-individual[me].surplus[myneed];

					if ((productionlimit>INTMULTIPLIER) &&
//						(((individual[me].purchasetimes[myneed] == 0) && individual[me].surplus[myneed]>(MAXSURPLUSFACTOR * market.basicneed[1]))||
						(1.0f / individual[me].efficiency[myneed] <= (PRICEHIKE * individual[me].salesprice[myneed])))				{
//							we try and produce surplus for those needs where our added value is greatest
//							WARNING!!!!! there are several possible heuristics, we compare now actualized profit margins
//							if (productionindex <= individual[me].efficiency[myneed] - (1.0f / individual[me].salesprice[myneed]))								{
//									productionindex = individual[me].efficiency[myneed] - (1.0f / individual[me].salesprice[myneed]);

							if (productionindex <= (individual[me].efficiency[myneed] -
								(1.0f / individual[me].salesprice[myneed])))								{
									productionindex = individual[me].efficiency[myneed] -
										(1.0f / individual[me].salesprice[myneed]);
									selectedproduction = myneed;
									productionlimitselectedproduction = productionlimit;
//									printf("productionindex %f myneed %i productionlimit %lli\n",productionindex, myneed, productionlimit);
							}
					}
				}
			}
			if (selectedproduction == 0) {//printf("selectedprodzero, but hello its me %i", me);
			break;}
			;
			if (me == TESTINDIVIDUAL)	printf("Me %i needtype %i Surplusproducing %i, productionlimitselectedproduction %lli surplus %lli efficiency %f timeremaining %lli, stocklimit %lli\n",
				me, needtype, selectedproduction, productionlimitselectedproduction, individual[me].surplus[selectedproduction],individual[me].efficiency[selectedproduction], individual[me].periodremaining, individual[me].stocklimit[selectedproduction]);

			maxproduction = (long long) (individual[me].efficiency[selectedproduction] * individual[me].periodremaining);
			if (maxproduction < productionlimitselectedproduction)
			{
				individual[me].surplus[selectedproduction] = individual[me].surplus[selectedproduction] + maxproduction;
				individual[me].producedthisperiod[selectedproduction] = individual[me].producedthisperiod[selectedproduction] + maxproduction;
				individual[me].recentlyproduced[selectedproduction] = individual[me].recentlyproduced[selectedproduction] + maxproduction;
				individual[me].periodremaining = 0; //note!!! all remaining time being used in this upper part for production
				if (me==TESTINDIVIDUAL) printf("if\n");
			}
				else {
					maxtimeforsurplus = SWITCHTIME + (long long)(productionlimitselectedproduction / individual[me].efficiency[selectedproduction]);

//					productionlimit = (long long)(maxtimeforsurplus * individual[me].efficiency[selectedproduction]);
					individual[me].surplus[selectedproduction] = individual[me].surplus[selectedproduction] + productionlimitselectedproduction;
					individual[me].producedthisperiod[selectedproduction] = individual[me].producedthisperiod[selectedproduction] + productionlimitselectedproduction;
					individual[me].recentlyproduced[selectedproduction] = individual[me].recentlyproduced[selectedproduction] + productionlimitselectedproduction;
					individual[me].periodremaining = individual[me].periodremaining - maxtimeforsurplus;
					if (me==TESTINDIVIDUAL) printf("else\n");
			}
			if (individual[me].periodremaining <0) printf("periodremaining negative");
			if (me == TESTINDIVIDUAL)
				printf("Produced surplus %i, efficiency %f timeremaining %lli, stocklimit %lli, surplus %lli, productionlimselprod %lli\n",
											selectedproduction, individual[me].efficiency[selectedproduction], individual[me].periodremaining,
											individual[me].stocklimit[selectedproduction], individual[me].surplus[selectedproduction], productionlimitselectedproduction);

	}

}

void leisureproduction(int me){
	int selectedproduction;
	long long productionlimit;
	long long productionlimitselectedproduction=0;
	long long maxproduction;
	float productionindex;
	float currentproductionindex; //not yet in use - production index may be multiplied by efficiency to favour investment in experience
	long long maxtimeforsurplus;

	while (individual[me].periodremaining >= 1)		{
			productionlimit =0;
			productionindex = 0;
			selectedproduction = 0;
			maxtimeforsurplus = 0;

			for (int myneed=1; myneed<SKILLS; myneed++)
			{
				if (individual[me].gifted[myneed] == FALSE)	{ //this subroutine is entered only when there is no room for skilled production in stock
					    productionlimit = individual[me].stocklimit[myneed]-individual[me].surplus[myneed];

					if ((productionlimit>INTMULTIPLIER) && (productionindex <= individual[me].purchaseprice[myneed])){
						productionindex = individual[me].purchaseprice[myneed];
						selectedproduction = myneed;
						productionlimitselectedproduction = productionlimit;
					}

				}
			}
			if (selectedproduction == 0) {//printf("selectedprodzero, but hello its me %i", me);
			break;}
			;
			if (me == TESTINDIVIDUAL)	printf("Me %i Leisureproducing %i, productionlimitselectedproduction %lli surplus %lli efficiency %f timeremaining %lli, stocklimit %lli\n",
				me, selectedproduction, productionlimitselectedproduction, individual[me].surplus[selectedproduction],individual[me].efficiency[selectedproduction], individual[me].periodremaining, individual[me].stocklimit[selectedproduction]);

			maxproduction = (long long) individual[me].periodremaining;
			if (maxproduction < productionlimitselectedproduction)
			{
				individual[me].surplus[selectedproduction] = individual[me].surplus[selectedproduction] + maxproduction;
				individual[me].producedthisperiod[selectedproduction] = individual[me].producedthisperiod[selectedproduction] + maxproduction;
				individual[me].recentlyproduced[selectedproduction] = individual[me].recentlyproduced[selectedproduction] + maxproduction;
				individual[me].periodremaining = 0; //note!!! all remaining time being used in this upper part for production
				if (me==TESTINDIVIDUAL) printf("zeroedleisuretime\n");
			}
				else {
					maxtimeforsurplus = productionlimitselectedproduction;

					individual[me].surplus[selectedproduction] = individual[me].surplus[selectedproduction] + productionlimitselectedproduction;
					individual[me].producedthisperiod[selectedproduction] = individual[me].producedthisperiod[selectedproduction] + productionlimitselectedproduction;
					individual[me].recentlyproduced[selectedproduction] = individual[me].recentlyproduced[selectedproduction] + productionlimitselectedproduction;
					individual[me].periodremaining = individual[me].periodremaining - maxtimeforsurplus;
					if (me==TESTINDIVIDUAL) printf("timeremaining\n");
			}
			if (individual[me].periodremaining <0) printf("periodremaining negative");
			if (me == TESTINDIVIDUAL)
				printf("Leisureproduced surplus %i, efficiency %f timeremaining %lli, stocklimit %lli, surplus %lli, productionlimselprod %lli\n",
											selectedproduction, individual[me].efficiency[selectedproduction], individual[me].periodremaining,
											individual[me].stocklimit[selectedproduction], individual[me].surplus[selectedproduction], productionlimitselectedproduction);

	}

}


void consumesurplus(int me) //here we satisfy our needs from possible surplus
{
		for (int j=1; j<SKILLS; j++)
		{
			if (individual[me].surplus[j] > individual[me].need[j])								{
					individual[me].surplus[j] = individual[me].surplus[j] - individual[me].need[j];
					individual[me].need[j] = 0;
				}
			else
			    if (individual[me].surplus[j] > 0)									{
					individual[me].need[j] = individual[me].need[j] - individual[me].surplus[j];
					individual[me].surplus[j] = 0;
				}
		if (individual[me].surplus[j] > (LLONG_MAX / MAXGROUP)) printf("Agent %i with surplus %i closes on llong_max /n",me, j);
		if (individual[me].surplus[j] < 0) {printf("consumesurplus negative, set to zero");individual[me].surplus[j] = 0;}
		}
}

void calibrateftransparency (int me)
{
	for (int k=1; k<MAXGROUP; k++)
	{
		for (int l=1; l<SKILLS; l++) //set transparency - presently transactions, efficiency and purchases are factored
			{

			individual[me].relationship[k].transparency[l] = INITIALSOCIALTRANSPARENCY;
			if (individual[me].relationship[k].transactions > 0)
				individual[me].relationship[k].transparency[l] = individual[me].relationship[k].transparency[l] +
				    (((1.0f - individual[me].relationship[k].transparency[l]) * 0.7f) *
				    ((float)individual[me].relationship[k].transactions /(float)(individual[me].relationship[k].transactions + SKILLS)));
			individual[me].relationship[k].transparency[l] = individual[me].relationship[k].transparency[l] +
			    (((1.0f - individual[me].relationship[k].transparency[l]) * 0.7f) *
			    (float)((10 * individual[me].relationship[k].purchased[l]) / (float)((10 * individual[me].relationship[k].purchased[l]) + market.period)));
			individual[me].relationship[k].transparency[l] = individual[me].relationship[k].transparency[l] +
			    ((((float)1 - individual[me].relationship[k].transparency[l]) * 0.7f) * //compiler error - (float) instead of 1.0f needed
			    (individual[me].recentlypurchased[l] / (individual[me].recentlypurchased[l] + ((10* HISTORY)*INTMULTIPLIER))));
			if (individual[me].gifted[l] == TRUE)
				individual[me].relationship[k].transparency[l] = individual[me].relationship[k].transparency[l] +
				    ((1.0f - individual[me].relationship[k].transparency[l]) * 0.5f);
		}
		if (individual[me].relationship[k].transactions > 1)
			individual[me].relationship[k].transactions = 0.9f * individual[me].relationship[k].transactions;
	}

}


void adjustpurchaseprice (int me, int myneed)
{
	// Let us now adjust purchaseprices - we use following heuristics for purchaseprice:

	// if I have more than periodic need surplus, I will not increase purchaseprice but if I have less
	// I will increase purchaseprice if it is lower than salesprice if there is sales or productioncost.
	// If I have maximum level of surplus and there were sales, I will lower purchaseprice
	// to what is suggested by averagepurchasevalue or if no purchases I will use general PRICEREDUCTION

		float productioncost = 1.0f / individual[me].efficiency[myneed];
//		float modpurchprice = (((productioncost * individual[me].recentlyproduced[myneed])+(individual[me].recentlysold[myneed]*individual[me].salesprice[myneed]))/(individual[me].recentlyproduced[myneed]+individual[me].recentlysold[myneed]));
//		if ((individual[me].recentlyproduced[myneed]+individual[me].recentlysold[myneed]) > individual[me].stocklimit[myneed]/2)
//			if (individual[me].purchaseprice[myneed] > modpurchprice)
//				individual[me].purchaseprice[myneed] = modpurchprice;

		switch ((individual[me].surplus[myneed] > (market.elasticneed[myneed])) +
				(individual[me].surplus[myneed] > (individual[me].stocklimit[myneed] + market.elasticneed[myneed]))) {

			case 0: //there is little surplus

				switch (individual[me].role[myneed]){
					case CONSUMER:
						if (individual[me].recentlypurchased[myneed] < individual[me].recentlyproduced[myneed]+INTMULTIPLIER)
							individual[me].purchaseprice[myneed] = productioncost;
						else if ((individual[me].purchasetimes[myneed]==0)&&(individual[me].purchaseprice[myneed] < productioncost))
							individual[me].purchaseprice[myneed] = PRICELEAP * individual[me].purchaseprice[myneed];
						break;
					case RETAILER:
						if ((individual[me].purchasedthisperiod[myneed] < individual[me].soldthisperiod[myneed] + INTMULTIPLIER) ||
							(individual[me].purchasedthisperiod[myneed] < individual[me].stocklimit[myneed]/HISTORY))
							individual[me].purchaseprice[myneed] = PRICEHIKE * individual[me].purchaseprice[myneed];
						break;
					case PRODUCER:
						break;
					default:
						printf("this is unused option in switch/case when finetuning sales and purchase prices");
				}
				break;

			case 1:  //there is average surplus
				switch (individual[me].role[myneed]){
					case CONSUMER:
						break;
					case RETAILER:
						if ((individual[me].purchasetimes[myneed]>1) && (individual[me].purchasedthisperiod[myneed] > (individual[me].stocklimit[myneed]/2)))
							{
							individual[me].purchaseprice[myneed] = ((individual[me].sumperiodpurchasevalue[myneed]+(HISTORY * individual[me].purchaseprice[myneed]))
																	/ (float)(individual[me].purchasetimes[myneed]+HISTORY));
							if(me==TESTINDIVIDUAL) printf("setting myneed %i  avpurchvalue %f  sumperpurchval %f sumperpurchtimes %i\n",
								myneed, individual[me].purchaseprice[myneed],individual[me].sumperiodpurchasevalue[myneed],individual[me].purchasetimes[myneed]);
							}

						if (individual[me].purchaseprice[myneed] > individual[me].salesprice[myneed])
							individual[me].purchaseprice[myneed] = (PRICEREDUCTION * individual[me].salesprice[myneed]);
						break;
					case PRODUCER:
							if (individual[me].purchasedthisperiod[myneed] > individual[me].producedthisperiod[myneed])
							individual[me].purchaseprice[myneed] = PRICEREDUCTION * individual[me].purchaseprice[myneed];
						break;
					default:
						printf("this is unused option in switch/case when finetuning sales and purchase prices");
				}
				break;

			case 2:  //there is a large surplus, average purchasevalue is calculated, but balanced with previous purchaseprice
				switch (individual[me].role[myneed]){
					case CONSUMER:
						individual[me].purchaseprice[myneed] = PRICEREDUCTION * individual[me].purchaseprice[myneed];
						break;
					case RETAILER:
						if (individual[me].purchasetimes[myneed])
							individual[me].purchaseprice[myneed] = PRICEREDUCTION * individual[me].purchaseprice[myneed];
						break;
					case PRODUCER:
						if (individual[me].purchasetimes[myneed])
							individual[me].purchaseprice[myneed] = PRICEREDUCTION * individual[me].purchaseprice[myneed];
						break;
					default:
						printf("this is unused option in switch/case when finetuning sales and purchase prices");
				}
				break;

			default:
				printf("this is unused option in switch/case when finetuning sales and purchase prices");

		}
		if (individual[me].purchaseprice[myneed] > 10) printf ("Role %i purchaseprice %f\n", individual[me].role[myneed],individual[me].purchaseprice[myneed]);
}

void adjustsalesprice(int me, int myneed)
{
	// Let us now adjust salesprices - we use following heuristics.

	//if I have sales and little or no surplus I will increase salesprice to what is suggested by statistics
	//and check that it is higher than the lower of purchaseprice and productioncost
	//if I have no sales and little or no surplus I will set salesprice to little over productioncost !!!!!
	//(if it is exactly productioncost, it will not be produced as it is considered useless waste
	//if I have sales and medium surplus I will increase sales to what is suggested by statistics
	//if no sales and large surplus I will reduce salesprice unless lower than both purchaseprice and purchaseprice

		float productioncost = 1.0f / individual[me].efficiency[myneed];
		float previoussalesprice = individual[me].salesprice[myneed];
//		float costofproduct = (((productioncost * individual[me].recentlyproduced[myneed])+(individual[me].recentlypurchased[myneed]*individual[me].purchaseprice[myneed]))/(individual[me].recentlyproduced[myneed]+individual[me].recentlypurchased[myneed]));
//		if (((individual[me].recentlyproduced[myneed]+individual[me].recentlypurchased[myneed]) > (individual[me].stocklimit[myneed]/2)) &&
//			(individual[me].salesprice[myneed] < costofproduct) && individual[me].salestimes[myneed] &&
//			(individual[me].recentlysold[myneed] > (individual[me].stocklimit[myneed]/2)))
//			individual[me].salesprice[myneed] = costofproduct;

		switch ((individual[me].surplus[myneed] > market.elasticneed[myneed]) +
				(individual[me].surplus[myneed] > (individual[me].stocklimit[myneed]+market.elasticneed[myneed]))) {

			case 0: //there is little surplus, if sales, salesprice can perhaps be raised - to average sales value
					//if purchases, we must raise salesprice to the lower of productioncost and purchaseprice if lower

				switch (individual[me].role[myneed]){
					case CONSUMER:
						//raise to smaller of productioncost or purchaseprice and add profit
						if (individual[me].salesprice[myneed] < (productioncost>individual[me].purchaseprice[myneed]?individual[me].purchaseprice[myneed]:productioncost)){
							individual[me].salesprice[myneed] = (productioncost>individual[me].purchaseprice[myneed]?individual[me].purchaseprice[myneed]:productioncost);
							individual[me].salesprice[myneed] = PRICEHIKE * individual[me].salesprice[myneed];
						}
						break;
					case RETAILER:
						if ((individual[me].salestimes[myneed]>1) && (individual[me].soldthisperiod[myneed] > (individual[me].stocklimit[myneed])/2))
						{
							individual[me].salesprice[myneed] = ((individual[me].sumperiodsalesvalue[myneed]+individual[me].salesprice[myneed]) / (float)(individual[me].salestimes[myneed]+1));
							if (individual[me].salesprice[myneed] > (PRICELEAP) * previoussalesprice)
								individual[me].salesprice[myneed] = (PRICELEAP) * previoussalesprice;
						}
						if (individual[me].salesprice[myneed] < individual[me].purchaseprice[myneed])
							individual[me].salesprice[myneed] = individual[me].purchaseprice[myneed];
						if (individual[me].salestimes[myneed] == 0)
							individual[me].salesprice[myneed] = PRICEHIKE * individual[me].purchaseprice[myneed];

						break;
					case PRODUCER:
						if (individual[me].salesprice[myneed] < productioncost)
							individual[me].salesprice[myneed] = PRICEHIKE*productioncost;
						break;
					default:
						printf("this is unused option in switch/case when finetuning sales and purchase prices");
				}
				break;

			case 1:  //there is average surplus, check salesprice  - if no sales, adjust salesprice
				switch (individual[me].role[myneed]){
					case CONSUMER:
						individual[me].salesprice[myneed] = (productioncost > individual[me].purchaseprice[myneed]?productioncost:individual[me].purchaseprice[myneed]);
						if (individual[me].surplus[myneed]>(individual[me].stocklimit[myneed]/2))
							individual[me].salesprice[myneed] = (((individual[me].purchaseprice[myneed]*2)<productioncost)?(individual[me].purchaseprice[myneed]*2):productioncost);
						break;
					case RETAILER:
						if (individual[me].soldthisperiod[myneed] < (market.elasticneed[myneed])){
//							individual[me].salesprice[myneed] = ;
							individual[me].salesprice[myneed] = individual[me].salesprice[myneed]*PRICEREDUCTION;
						}
						break;
					case PRODUCER:
						if (individual[me].soldthisperiod[myneed] < (market.elasticneed[myneed])){
//							individual[me].salesprice[myneed] = productioncost * PRICEHIKE;
							individual[me].salesprice[myneed] = individual[me].salesprice[myneed]*PRICEREDUCTION;
						}
						if (individual[me].salesprice[myneed] < productioncost)
							individual[me].salesprice[myneed] = PRICEHIKE * productioncost;


						break;
					default:
						printf("this is unused option in switch/case when finetuning sales and purchase prices");
				}


				break;

			case 2:  //there is a large surplus, discount is necessary and if no sales,
					 //salesprice is set to productioncost directly unless purchasevolume is considerable
				switch (individual[me].role[myneed]){
					case CONSUMER:
							individual[me].salesprice[myneed] = individual[me].salesprice[myneed]*PRICEREDUCTION;
						break;
					case RETAILER:
						if (individual[me].soldthisperiod[myneed] < (market.elasticneed[myneed])){
							if (individual[me].salesprice[myneed] > individual[me].purchaseprice[myneed])
								individual[me].salesprice[myneed] = individual[me].purchaseprice[myneed];
							else
							individual[me].salesprice[myneed] = individual[me].salesprice[myneed]*PRICEREDUCTION;
						}
						break;
					case PRODUCER:
						if (individual[me].salesprice[myneed] > productioncost)
								individual[me].salesprice[myneed] = PRICEHIKE * productioncost;
						if (individual[me].soldthisperiod[myneed] < (market.elasticneed[myneed])){
						individual[me].salesprice[myneed] = productioncost * PRICEHIKE;
//						individual[me].salesprice[xmyneed] = individual[me].salesprice[myneed]*PRICEREDUCTION;
					}
						break;
					default:
						printf("this is unused option in switch/case when finetuning sales and purchase prices");
				}
				break;

			default:
				printf("this is unused option in switch/case when finetuning sales and purchase prices");
		}
}

float evaluatestock (int me){ //preliminary calculation using our price expectations to see if our needslevel is sustainable
		float stocklevel=1.0f;
		double wealthminusneeds = market.periodlength;
		double surplusvalue = 0.0f;
		double totalneedsvalue = 0.0f;

		for (int need=1; need < SKILLS; need++){
			surplusvalue = individual[me].surplus[need] - ((market.elasticneed[need]*individual[me].needslevel)*MAXNEEDSINCREASE);
			if (surplusvalue < 0) {
				if (individual[me].purchasedxperiod[need] > -1 * surplusvalue)
					surplusvalue = surplusvalue * individual[me].purchaseprice[need];
				else surplusvalue = surplusvalue / individual[me].efficiency[need];
				}
				else {
					if (individual[me].recentlysold[need] > surplusvalue){ // if unable to sell, we do not count it
						if (surplusvalue > MAXSURPLUSFACTOR * ((individual[me].soldthisperiod[need] - individual[me].soldxperiod[need])+(market.elasticneed[need]*individual[me].needslevel)))
							surplusvalue = MAXSURPLUSFACTOR * ((individual[me].soldthisperiod[need] - individual[me].soldxperiod[need])+(market.elasticneed[need]*individual[me].needslevel));
						surplusvalue = surplusvalue * individual[me].salesprice[need];
					}
					else {
						if (surplusvalue > ((market.elasticneed[need]*individual[me].needslevel)*MAXNEEDSINCREASE))
							surplusvalue = (market.elasticneed[need]*individual[me].needslevel)*MAXNEEDSINCREASE;
						surplusvalue = surplusvalue * (individual[me].purchaseprice[need]<(1.0f/individual[me].efficiency[need])?individual[me].purchaseprice[need]:(1.0f/individual[me].efficiency[need]));
					}
				}
			wealthminusneeds = wealthminusneeds + surplusvalue;
			if (individual[me].recentlypurchased[need] > (market.elasticneed[need]*individual[me].needslevel))
				totalneedsvalue = totalneedsvalue + (individual[me].purchaseprice[need]*(market.elasticneed[need]*individual[me].needslevel));
				else totalneedsvalue = totalneedsvalue + ((market.elasticneed[need]*individual[me].needslevel) / individual[me].efficiency[need]);
		}
		stocklevel = (totalneedsvalue + wealthminusneeds)/(totalneedsvalue);
		return stocklevel;
}


void endmyperiod (int me) //Rewind, update production cost, sales and purchase prices, efficiency and transparency
{

/*	//if there is much time left due to increased efficiency, consumption is increased through another round of spending and producing
	individual[me].recentneedsincrement = (individual[me].needsincrement + (HISTORY * individual[me].recentneedsincrement))/(float)(HISTORY + 1);
	individual[me].needsincrement = 0;
//	long long currentneedsincrement = 0;
	if (individual[me].periodremaining > market.leisuretime) {
		individual[me].needsincrement = -1.0f + ((double)(market.periodlength) / (double)(market.periodlength + market.leisuretime - individual[me].periodremaining));
		if (individual[me].needsincrement > ((individual[me].recentneedsincrement +1.0f) * MAXNEEDSINCREMENT)-1.0f)
			individual[me].needsincrement = ((individual[me].recentneedsincrement +1.0f) * MAXNEEDSINCREMENT)-1.0f;

		if (me == TESTINDIVIDUAL) printf("\n RERUN!! needs %f, recently %f periodremaining %lli \n",(1.0f + individual[me].needsincrement), (1.0f + individual[me].recentneedsincrement), individual[me].periodremaining);
		;
		float needsbuffer = individual[me].needsincrement;
		long long originalperiodremaining = individual[me].periodremaining;
		while (needsbuffer > 0.1f){
			needsbuffer = needsbuffer* SPENDINGFROMEXCESS;
			for (int myneed=1; myneed<SKILLS; myneed++) //If more than normal leisure time remaining, create new needs and continue an extra round
				individual[me].need[myneed] = (long long)(market.elasticneed[myneed] * needsbuffer);

			consumesurplus(me);
			makesurplusdeals(me);
			producesmallneeds(me);
			satisfyneedsbyexchange(me);
			produceneed(me);
			if (individual[me].periodremaining < (originalperiodremaining * SPENDINGFROMEXCESS)) needsbuffer=0;
			if (me == TESTINDIVIDUAL) printf("periodremaining %lli needsbuffer %.3f originalperiodsrem %lli SPENDINGFRX %f\n",individual[me].periodremaining, needsbuffer, originalperiodremaining, SPENDINGFROMEXCESS);
		}

		surplusproduction(me, SURPLUSROUND);
		checksurplus(10);
	} */
//	individual[me].periodremainingdebt = (individual[me].periodremainingdebt + individual[me].periodremaining);
//	if	(individual[me].periodremainingdebt > 0)
//		individual[me].periodremainingdebt = 0; //periodremainingdebt might also contain positive value??

	individual[me].periodicspoils = 0;
	for (int myneed=1; myneed<SKILLS; myneed++)
	{
		//Two options - we can have basicround with basicneed, but our statistics are then harder to interpret
		//or we can have all rounds with elasticneed, interpreting a needs shift - total number of items is same
		//and time required to produce them with initial efficiency is affected only through giftedness
//		individual[me].need[myneed] = market.elasticneed[myneed];
//		if (BASICROUNDELASTIC) individual[me].need[myneed] = market.elasticneed[myneed];
//		individual[me].need[myneed] = market.elasticneed[myneed];

		//if we require that basicneed is minimum and lower only for surplusrounds, we create a problem with lower income people
		//who do not produce surplus but run out of timeslots and poeriodremaining becomes negative with total surplus being zero
		//this requires further thought but currently it is interpreted that elasticity actually diminishes the more common
		//needs when their relative prices are higher and this actually might be true that to some extent needs are replaced by others
//		individual[me].need[myneed] = ((market.basicneed[myneed]>market.elasticneed[myneed])?market.basicneed[myneed]:market.elasticneed[myneed]);
//		individual[me].stocklimit[myneed] = (MAXSURPLUSFACTOR * (market.basicneed[myneed] + (individual[me].recentneedsincrement * market.elasticneed[myneed])) + individual[me].recentlysold[myneed]);
		individual[me].stocklimit[myneed] =
			((MAXSURPLUSFACTOR * (market.elasticneed[myneed]* individual[me].needslevel))+ individual[me].recentlysold[myneed]);
//		individual[me].stocklimit[myneed] = (MAXSURPLUSFACTOR * market.basicneed[myneed] + individual[me].recentlysold[myneed]);
		if (individual[me].stocklimit[myneed] < (MAXSTOCKLIMITDECREASE * individual[me].previousstocklimit[myneed]))
 			individual[me].stocklimit[myneed] = (MAXSTOCKLIMITDECREASE * individual[me].previousstocklimit[myneed]);
		if (individual[me].stocklimit[myneed] > (MAXSTOCKLIMITINCREASE * individual[me].previousstocklimit[myneed]))
 			individual[me].stocklimit[myneed] = (MAXSTOCKLIMITINCREASE * individual[me].previousstocklimit[myneed]);
		individual[me].previousstocklimit[myneed] = individual[me].stocklimit[myneed];


		if (individual[me].gifted[myneed] == TRUE)						{
				float previousefficiency = individual[me].efficiency[myneed];

				individual[me].efficiency[myneed] = (1.0f / InvSqrt((float)(individual[me].recentlyproduced[myneed]+1) /
					(float)(HISTORY * market.basicneed[myneed]))); //Efficiency increased by surplus production
				if (individual[me].efficiency[myneed] < (previousefficiency * MAXEFFICIENCYDOWNGRADE))
					individual[me].efficiency[myneed] = (previousefficiency * MAXEFFICIENCYDOWNGRADE);
					else if (individual[me].efficiency[myneed] > (previousefficiency * MAXEFFICIENCYUPGRADE))
						individual[me].efficiency[myneed] = (previousefficiency * MAXEFFICIENCYUPGRADE);
				if (individual[me].efficiency[myneed] < GIFTEDEFFICIENCYMINIMUM)
					individual[me].efficiency[myneed] = GIFTEDEFFICIENCYMINIMUM; //Note - unused gifted efficiency boost decreases to 50% initial value
			} else if ((me == TESTINDIVIDUAL) && (individual[me].recentlysold[myneed] > (market.elasticneed[myneed])))
						printf("Nongifted trading  --- myneed %i rsales %lli salesprice %f rpurchases %lli purchaseprice %f stocklimit %lli surplus %lli\n",
								myneed, individual[me].recentlysold[myneed], individual[me].salesprice[myneed], individual[me].recentlypurchased[myneed],
								individual[me].purchaseprice[myneed], individual[me].stocklimit[myneed], individual[me].surplus[myneed]);
checksurplus(11);
		//Redefine our role as consumer, producer or retailer - this affects heuristics in production, exchange and adjusting privatevalues
		individual[me].role[myneed] = CONSUMER; //Default is consumer
		if ((individual[me].recentlyproduced[myneed] > individual[me].recentlypurchased[myneed]) &&
			((individual[me].recentlysold[myneed]> (individual[me].recentlyproduced[myneed] + individual[me].recentlypurchased[myneed])/2)))
			individual[me].role[myneed] = PRODUCER; //Producer we are if over 50% of what we get is sold and over 50% is produced
		else if((individual[me].recentlyproduced[myneed] < individual[me].recentlypurchased[myneed]) &&
			((individual[me].recentlysold[myneed]> (individual[me].recentlyproduced[myneed] + individual[me].recentlypurchased[myneed])/2)))
			individual[me].role[myneed] = RETAILER; //Retailer we are if over 50% of what we get is sold and over 50% is purchased
		adjustpurchaseprice(me, myneed);
		adjustsalesprice(me,myneed);

		if (me==TESTINDIVIDUAL){
			printf("Need %i, role %i srplus %6lli, stcklmt %6lli, eff %3.1f, rprdcd %7lli prod %7lli rpurch %5lli purch %5lli purchtimes %i avpurchval %.3f p$ %.2f rsls %5lli sold %5lli sp$ %.2f\n",
			myneed, individual[TESTINDIVIDUAL].role[myneed],individual[TESTINDIVIDUAL].surplus[myneed], individual[TESTINDIVIDUAL].stocklimit[myneed],individual[TESTINDIVIDUAL].efficiency[myneed],
			individual[TESTINDIVIDUAL].recentlyproduced[myneed], individual[TESTINDIVIDUAL].producedthisperiod[myneed],
			individual[TESTINDIVIDUAL].recentlypurchased[myneed], individual[TESTINDIVIDUAL].purchasedthisperiod[myneed], individual[TESTINDIVIDUAL].purchasetimes[myneed],
			(individual[TESTINDIVIDUAL].purchasetimes[myneed]?(float)(individual[TESTINDIVIDUAL].sumperiodpurchasevalue[myneed]/individual[TESTINDIVIDUAL].purchasetimes[myneed]):0.0f),
			individual[TESTINDIVIDUAL].purchaseprice[myneed],individual[TESTINDIVIDUAL].recentlysold[myneed], individual[TESTINDIVIDUAL].soldthisperiod[myneed],
			individual[TESTINDIVIDUAL].salesprice[myneed]);
		}

		//Excess surplus is reduced as part of it is considered spoiled due to too slow turnaround
		individual[me].spoils[myneed] = 0;
		if (individual[me].surplus[myneed] > (MAXSURPLUSFACTOR * market.elasticneed[myneed]))
			//!!!!!!NOTE spoiled surplus starts from STOCKSPOILTRESHOLD
			if (individual[me].surplus[myneed] > (STOCKSPOILTRESHOLD*individual[me].stocklimit[myneed])) {
				if (individual[me].surplus[myneed] > (individual[me].stocklimit[myneed])) {
					individual[me].spoils[myneed] = (long long)((individual[me].surplus[myneed] - individual[me].stocklimit[myneed]) * SPOILSURPLUSEXCESS);
					individual[me].surplus[myneed] = individual[me].surplus[myneed] - individual[me].spoils[myneed];
					individual[me].periodicspoils = individual[me].periodicspoils + individual[me].spoils[myneed];
					market.periodicspoils[myneed] = market.periodicspoils[myneed] + individual[me].spoils[myneed];

					if ((individual[me].surplus[myneed] < individual[me].spoils[myneed])||(individual[me].spoils[myneed]<0))
						printf("Spoils-calculation overflow need %i, surplus %lli spoils %lli elasticneed %lli, recentlysold %lli???",
						myneed,individual[me].surplus[myneed],individual[me].spoils[myneed], (long long) market.elasticneed[myneed], individual[me].recentlysold[myneed]);
				}
				if (individual[me].surplus[myneed] < 0)
					printf("surplus negative, need %i surplus %lli", myneed, individual[me].surplus[myneed]);
				if (individual[me].periodicspoils < 0)
					printf("periodicspoils negative, indy %i periodic %lli need%i spoils %lli\n",me, individual[me].periodicspoils, myneed, individual[me].spoils[myneed]);
		}
		//Older transactions need to be discounted in order to emphasize recent events
		individual[me].recentlyproduced[myneed] = (long long)(DISCONT * (double) individual[me].recentlyproduced[myneed]);
		individual[me].recentlysold[myneed] = (long long)(DISCONT * (double) individual[me].recentlysold[myneed]);
		individual[me].recentlypurchased[myneed] = (long long)(DISCONT * (double)individual[me].recentlypurchased[myneed]);
checksurplus(12);
		//We reset periodic counters here after using the data as they are updated by other agents besides us
		individual[me].purchasetimes[myneed] = 0;
		individual[me].salestimes[myneed] = 0;
		individual[me].sumperiodpurchasevalue[myneed] = 0;
		individual[me].sumperiodsalesvalue[myneed] = 0;
		individual[me].producedxperiod[myneed] = individual[me].producedthisperiod[myneed];
		individual[me].soldxperiod[myneed] = individual[me].soldthisperiod[myneed];
		individual[me].purchasedxperiod[myneed] = individual[me].purchasedthisperiod[myneed];
	}
	calibrateftransparency(me);
}

/*void resetperiodicstat (void){
		for (int i=1;i<POPULATION; i++){
				for (int need=1; need < SKILLS; need++){
			individual[i].producedthisperiod[need] = 0;
			individual[i].soldthisperiod[need] = 0;
			individual[i].purchasedthisperiod[need] = 0;


			}
		}
		for (int need=1; need < SKILLS; need++){
			market.periodictcecost[need] = 0;
			market.periodicspoils[need] =0;

		}
		for (int i=1; i<SKILLS; i++) {
		}

}*/
void evolution (void){	//first satisfy needs from your own surplus
//	printf("evolution menossa \n");
	for (int i=1;i<POPULATION; i++)
	{
		//now we are ready to satisfy unsatisfied needs, first rewind the periodclock, other counters rewound
		//after previous round and they reflect now exchanges that other agents have activated in the meantime

		if ((individual[i].periodremainingdebt < -2*market.periodlength) || (individual[i].periodremaining > (2*market.periodlength))){
			printf("periodremaining overflow, me %i, needslevel %.3f periods %lli, debt %lli \n",
					i, individual[i].needslevel, individual[i].periodremaining, individual[i].periodremainingdebt);
			market.testindividual = i;
		}
		individual[i].periodremaining = market.periodlength;

		for (int need=1; need < SKILLS; need++){
			individual[i].producedthisperiod[need] = 0;
			individual[i].soldthisperiod[need] = individual[i].soldthisperiod[need] - individual[i].soldxperiod[need];
			individual[i].purchasedthisperiod[need] = individual[i].purchasedthisperiod[need] - individual[i].purchasedxperiod[need];
//			if (individual[i].producedthisperiod[need]==0);
//			else	printf("me %i producednow %i - predicted to be zero %lli\n",i, need, individual[i].producedthisperiod[i]);
		}
		float stocklevel=0.0f;
		stocklevel=evaluatestock(i);
		if ((stocklevel < MAXNEEDSREDUCTION) || individual[i].periodfailure) //we check if our stock value can sustain our needslevel
			individual[i].needslevel = individual[i].needslevel * MAXNEEDSREDUCTION;
			else if ((individual[i].periodremainingdebt > ((1.0f-MAXNEEDSINCREASE)*market.periodlength))&& (stocklevel>MAXNEEDSINCREASE))
				individual[i].needslevel = MAXNEEDSINCREASE * individual[i].needslevel;
					else if (stocklevel > SMALLNEEDSINCREASE)
							individual[i].needslevel = individual[i].needslevel * SMALLNEEDSINCREASE;
							else individual[i].needslevel = individual[i].needslevel * SMALLNEEDSREDUCTION;


		if (i==TESTINDIVIDUAL) printf("Kilroy %i was here and stocklevel is %.3f, needslevel is %f, periodleft is %lli and debt is %lli!\n",
	   		 i, stocklevel, individual[i].needslevel, individual[i].periodremaining, individual[i].periodremainingdebt);
		if (individual[i].needslevel < 1) individual[i].needslevel = 1;
/*		while (individual[i].needslevel > 0.99) {
			if (i==TESTINDIVIDUAL) printf("Kilroy %i was here and stocklevel is %.3f!", i, stocklevel);
			if (satisfyneedsbyexchange(i)){
			    break;
			}
			individual[i].needslevel=individual[i].needslevel * NEEDSREDUCTION;
		}
		if (individual[i].needslevel < 1) individual[i].needslevel = 1;
*/
		if (individual[i].periodremainingdebt < (-1 * market.periodlength)){
//			individual[i].needslevel = (individual[i].needslevel / 2);

			individual[i].needslevel = 1.0f;
			printf("Kilroy %i starts from scratch, ends did not meet - needslevel dropped to 1, part of debts collected each round!\n", i);
		}

		for (int need=1; need < SKILLS; need++){
			if (individual[i].needslevel >= SMALLNEEDSINCREASE)
				individual[i].need[need] = market.elasticneed[need] * individual[i].needslevel;
//				else individual[i].need[need] = market.basicneed[need];
				else individual[i].need[need] = market.elasticneed[need];
		}
		consumesurplus(i);
		satisfyneedsbyexchange(i);

		produceneed(i);
		if (individual[i].periodremaining < 0)
			individual[i].periodfailure = 1;
		else individual[i].periodfailure = 0;
//		if (individual[i].periodremaining < -1 * (market.periodlength * (1.0f-NEEDSREDUCTION)))
//			individual[i].needslevel = 1;
		makenewfriends(i, FALSE);
		if (individual[i].periodremainingdebt < 0){
			individual[i].periodremaining = individual[i].periodremaining + (individual[i].periodremainingdebt/2);
			individual[i].periodremainingdebt = (individual[i].periodremainingdebt/2);
		} //part of periodremaining is used to cut down debt;
		surplusproduction(i, SURPLUSROUND);
		makesurplusdeals(i);
		individual[i].periodremainingdebt = individual[i].periodremaining + individual[i].periodremainingdebt;
		if	(individual[i].periodremainingdebt > 0)
		individual[i].periodremainingdebt = 0; //periodremainingdebt might also contain positive value??
//		printf("starting leisureproduction");
		leisureproduction(i);

/*		consumesurplus(i); //first we fill our needs from our own surplus
		//now we check wether we can satisfy unsatisfied needs through exchange to our remaining surplus
		checksurplus(2);
		satisfyneedsbyexchange(i);
		//for needs that were not satisfied through exchange we use time to satisfy ourselves
		checksurplus(3);
		produceneed(i);
		//now we are ready to make one new social aquintance
		checksurplus(4);
		makenewfriends(i, FALSE);
		//remaining time we use to produce surplus but not more than previous history shows to be useful in very near future
		surplusproduction(i, REGULARROUND);
		checksurplus(5);
*/		//if there is ample remaining time, our needs are increased temporarily and we rerun previous steps
		//and then with periodic experience we evaluate sales and purchase prices and transparency to friends
		endmyperiod(i);
	}
}


int main(int argc,char *argv[]){
	FILE *f = fopen(FILENAME,"w");
	fprintf(f, "period, utility, spoils, tcecost, stored, production, needsdeviation, dev/aveinc, prod1maxIndy, overtime, pprice1, pp1dev, sumprod1, tcepertime, spoilspertime, maxef1*maxef1, totneed1inc\n");
	fprintf(f, "AdamSmith - population %i group %i skills %i transparency %.2f intmultiplier %i\n\n", POPULATION, MAXGROUP, SKILLS, INITIALSOCIALTRANSPARENCY, INTMULTIPLIER);
	printf("hey Adam Smith, it's Jan21 2011!\n");
	initmarket();
	initpopulation();
	selectgiftedpopulation();
	market.totalmisurplus = 0;
	for (int i=1; i<POPULATION;i++){
		for (int need=1; need < SKILLS; need++){
			individual[i].producedxperiod[need] = market.basicneed[need];
			individual[i].soldxperiod[need] = 0;
			individual[i].purchasedxperiod[need] = 0;
		}
	}
	for (int j=1; j < SKILLS; j++)
	{
		if (individual[TESTINDIVIDUAL].gifted[j] == TRUE) printf("indy is gifted for need %i \n",j);
	}

	printf("evoluutio alkaa\n");
	for (int period=1; period < MAXPERIODS; period++)
	{
//		resetperiodicstat();
		for (int me=1; me<POPULATION; me++){
			individual[me].totalsurplus = 0;
		}
		for (int need=1; need < SKILLS; need++){
			market.periodictcecost[need] = 0;
			market.periodicspoils[need] =0;
		}
		market.period = period;
		evolution();
		evaluatemarketprices(); //this subroutine produces most online reports and the rest are in the end of this main program
		long long totalproduction = 0;
		long long totalperiodictcecost = 0;

		for (int i = 1; i < SKILLS; i++) {
//			market.totalsurplus = market.totalsurplus + market.surplus[i];
			totalproduction = totalproduction + market.producedthisperiod[i];
			totalperiodictcecost = totalperiodictcecost + market.periodictcecost[i];
//			market.recentlypurchased [i] = (long long)((double)market.recentlypurchased [i] * DISCONT);
//			market.recentlysold [i] = (long long)((double)market.recentlysold [i] * DISCONT);
			market.numberofrecentlyproduced[i] = (long long)((double)market.numberofrecentlyproduced[i] * DISCONT);

			printf("Need %i, role %i srplus %6lli, stcklmt %6lli, eff %3.1f, rprdcd %7lli prodnow %7lli rpurch %6lli, purchnow %6lli p$ %.2f rsls %6lli soldnow %6lli sp$ %.2f\n",
			i, individual[TESTINDIVIDUAL].role[i],individual[TESTINDIVIDUAL].surplus[i], individual[TESTINDIVIDUAL].stocklimit[i],
			individual[TESTINDIVIDUAL].efficiency[i],individual[TESTINDIVIDUAL].recentlyproduced[i], individual[TESTINDIVIDUAL].producedthisperiod[i],
			individual[TESTINDIVIDUAL].recentlypurchased[i], individual[TESTINDIVIDUAL].purchasedthisperiod[i],
			individual[TESTINDIVIDUAL].purchaseprice[i],individual[TESTINDIVIDUAL].recentlysold[i], individual[TESTINDIVIDUAL].soldthisperiod[i],
			individual[TESTINDIVIDUAL].salesprice[i]);

			}
		if (totalperiodictcecost > (LLONG_MAX / MAXGROUP)) printf("totalperiodictcecost closes on llongmax");
		if (totalproduction > (LLONG_MAX / MAXGROUP)) printf("totalproduction closes on llongmax");

		double sumremainingperiods = 0;
		long long totalisurplus = 0;
		long long sumofspoils = 0;
		float sumovertimes =0.0f;
		double sumneeds = 0.0f;
		double sumrecentneeds = 0.0f;
		double sumpurchaseprice1 = 0.0f;
		market.loosers = 0;
		for (int i=1; i<POPULATION; i++) {
			totalisurplus = totalisurplus + individual[i].totalsurplus;
			sumremainingperiods = sumremainingperiods + individual[i].periodremaining;
//			sumneedsincrement = sumneedsincrement + individual[i].needsincrement;
			sumneeds = sumneeds + individual[i].needslevel;
			sumrecentneeds = sumrecentneeds + individual[i].recentneedsincrement;
			sumofspoils = sumofspoils + individual[i].periodicspoils;
			sumovertimes = sumovertimes + individual[i].timeout;
			if (individual[i].periodremainingdebt < (-1 * market.periodlength)) {
//				market.testindividual = i; //!!!!!!TESTINDIVIDUAL
				market.loosers++;
			}
			individual[i].timeout=0; //reset individual timeout after calculating sum value
			sumpurchaseprice1 = sumpurchaseprice1 + individual[i].purchaseprice[1];

		}

//!!!!!!!!!!Recentneedsdeviation not based on needslevel - must be corrected
		if (totalisurplus > (LLONG_MAX / MAXGROUP)) printf("totalisurplus closes on llongmax");
		double recentneedsdeviation = 0;
		float indydeviation = 0.0f;
		float averageneeds = sumneeds / POPULATION;
		float averagerecentneeds = sumrecentneeds / POPULATION;
		for (int i=1; i<POPULATION; i++){ //NOTE - we use wrongly periodic average and compound individual data but error should not be large
			indydeviation = (averagerecentneeds - individual[i].recentneedsincrement);
			recentneedsdeviation = recentneedsdeviation + ((indydeviation < 0)? -indydeviation : indydeviation);
		}
		recentneedsdeviation = recentneedsdeviation / POPULATION;

		double sumpurchase1deviation = 0.0f;
		float purchaseprice1average = sumpurchaseprice1 / POPULATION;
		float averagepurchaseprice1deviation = 0.0f;
		for (int i=1; i<POPULATION; i++){ //NOTE - we use wrongly periodic average and compound individual data but error should not be large
			sumpurchase1deviation = sumpurchase1deviation + (double)fabs(purchaseprice1average - individual[i].purchaseprice[1]);
		}
		averagepurchaseprice1deviation = (float)(sumpurchase1deviation / POPULATION);

		printf("\np%i freetime %.3f utility %.3f spoilitptime %.4f tcecostitptime %.4f storeditptime %.3f totproditptime %.3f prodbalance %.3f\nloosers %i needsdeviation %.4f dev/aveinc %.4f maxneed1satisfied %.4f overtime %.3f tcepertime %.3f spoilstime %.3f\n\n",
				period, (float)(sumremainingperiods / (POPULATION *market.periodlength)), (averageneeds),
				(float)sumofspoils/(market.periodlength*POPULATION),
				(float)((double)totalperiodictcecost/ (POPULATION*market.periodlength)),
				(float)((double)(totalisurplus - market.totalmisurplus)/(POPULATION*market.periodlength)),
				(float)((double)totalproduction /(market.periodlength*POPULATION)),
				((float)((double)totalproduction /(market.periodlength*POPULATION))-
				(((float)((double)(totalisurplus - market.totalmisurplus)/(POPULATION*market.periodlength))+
				(float)sumofspoils/(market.periodlength*POPULATION)+(averageneeds))+
				(float)((double)totalperiodictcecost/ (POPULATION*market.periodlength)))),
				market.loosers, (float)recentneedsdeviation, (float)(recentneedsdeviation / averagerecentneeds),
				(float)(market.maxefficiency[1] * market.maxefficiency[1]*market.basicneed[1]) / ((averageneeds) * market.elasticneed[1]), sumovertimes/POPULATION,
				(float)market.totalcostoftceintime/(market.periodlength*POPULATION), (float)market.totalcostofspoilsintime / (market.periodlength*POPULATION));
		fprintf(f,"%i, %.3f, %.3f, %.4f, %.4f, %.3f, %.3f, %.4f, %.4f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n",
				period, (averageneeds), (float)((double)sumofspoils/POPULATION)/market.periodlength,
				(float)((double)totalperiodictcecost/ (POPULATION*market.periodlength)), (float)((double)(totalisurplus - market.totalmisurplus)/(POPULATION*market.periodlength)),
				(float)((double)totalproduction /(market.periodlength*POPULATION)), (float)recentneedsdeviation, (float)(recentneedsdeviation / (averagerecentneeds)) ,
				(float)(market.maxefficiency[1] * market.maxefficiency[1]) / ((averageneeds) * (market.priceaverage/market.averageprice[1])*(market.priceaverage/market.averageprice[1])), sumovertimes/POPULATION,
				purchaseprice1average, averagepurchaseprice1deviation, (float)market.numberofrecentlyproduced[1]/(POPULATION*market.basicneed[1]*(HISTORY+1), (float)(market.maxefficiency[1] * market.maxefficiency[1]), (float)((averageneeds) * (market.priceaverage/market.averageprice[1])*(market.priceaverage/market.averageprice[1]))),
				(float)market.totalcostoftceintime / (market.periodlength*POPULATION), (float)market.totalcostofspoilsintime/(market.periodlength*POPULATION),
				(float)(market.maxefficiency[1] * market.maxefficiency[1]), ((averageneeds) * (market.priceaverage/market.averageprice[1])*(market.priceaverage/market.averageprice[1])));//consider elasticneed...
		market.totalmisurplus = totalisurplus;// + market.totalsurplus;
		if (mode != '0') mode = getchar();

	}
	fclose(f);
	return 0;
}
