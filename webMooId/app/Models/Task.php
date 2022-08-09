<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Task extends Model
{
    // Table Name
    protected $table = 'task';
    // Primary Key
    public $primary_key = 'id';
    // Timestamps
    public $timestamps = false;

    protected $fillable = [
        'no_ktp',
        'nama_pekerjaan',
        'status',
        'waktu'
    ];

    public function petugas()
    {
        return $this->belongsTo('App\Models\Petugas','no_ktp');
    }
}
