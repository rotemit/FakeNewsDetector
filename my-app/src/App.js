import React from 'react'
import './App.css';

import {HomePage} from './Pages/HomePage'
import {EnterProfile} from './Pages/EnterProfile'
import {ScanPost} from './Pages/ScanPost'

import {Login} from './Pages/Login'
import { Route, Switch, withRouter} from "react-router-dom";

function App() {
    return (
      <Switch>
          <Route exact path='/' component={HomePage} />
          <Route path='/EnterProfile' component={EnterProfile} />
          <Route path='/ScanPost' component={ScanPost} />
          <Route path='/Login' component={Login} />
         
      </Switch>
    );
}


export default withRouter(App);
