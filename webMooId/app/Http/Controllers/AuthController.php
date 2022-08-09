<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class AuthController extends Controller
{
    public function login() {
        $data = array(
            'title' => 'Login'
        );
        return view('auth.pages.login')->with($data);
    }

    public function register() {
        $data = array(
            'title' => 'Register'
        );
        return view('auth.pages.register')->with($data);
    }
}
